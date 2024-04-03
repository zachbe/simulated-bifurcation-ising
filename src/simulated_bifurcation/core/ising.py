"""
Implementation of the Ising class.

Ising is an interface to the Simulated Bifurcation algorithm and is
used for optimizing user-defined polynomial. See models.Ising for
an implementation of the Ising model which behaves like other models
and polynomials.

See Also
--------
models.Ising:
    Implementation of the Ising model which behaves like other models and
    polynomials.
QuadraticPolynomial:
    Class to implement multivariate quadratic polynomials from SymPy
    polynomial expressions or tensors that can be casted to Ising model
    for Simulated Bifurcation algorithm compatibility purposes.

"""


from typing import Optional, TypeVar, Union, List

import torch
from numpy import ndarray
import numpy as np
import time

from ..optimizer import SimulatedBifurcationEngine, SimulatedBifurcationOptimizer

import ctypes
import os
import warnings
from time   import sleep

# Workaround because `Self` type is only available in Python >= 3.11
SelfIsing = TypeVar("SelfIsing", bound="Ising")


class Ising:

    """
    Internal implementation of the Ising model.

    Solving an Ising problem means finding a spin vector `s` (with values
    in {-1, 1}) such that, given a matrix `J` with zero diagonal and a
    vector `h`, the following quantity - called Ising energy - is minimal
    (`s` is then called a ground state):
    `-0.5 * ΣΣ J(i,j)s(i)s(j) + Σ h(i)s(i)`
    or `-0.5 x.T J x + h.T x in matrix notation.

    Parameters
    ----------
    J: (M, M) Tensor
        Square matrix representing the quadratic part of the Ising model
        whose size is `M` the dimension of the problem.
    h: (M,) Tensor | None, optional
        Vector representing the linear part of the Ising model whose size
        is `M` the dimension of the problem. If this argument is not
        provided (`h is None`), it defaults to the null vector.
    dtype: torch.dtype, default=torch.float32
        Data-type used for storing the coefficients of the Ising model.
    device: str | torch.device, default="cpu"
        Device on which the instance is located.

    Attributes
    ----------
    dtype
    device
    dimension : int
        Size of the Ising problem, i.e. number of spins.
    computed_spins : (A, M) Tensor | None
        Spin vectors obtained by minimizing the Ising energy. None if no
        solving method has been called.
    J: (M, M) Tensor
        Square matrix representing the quadratic part of the Ising model
        whose size is `M` the dimension of the problem.
    h: (M,) Tensor
        Vector representing the linear part of the Ising model whose size
        is `M` the dimension of the problem.
    linear_term: bool
        Whether the model has a non-zero linear term.

    See Also
    --------
    models.Ising:
        An implementation of the Ising model which behaves like other
        models and polynomials.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Ising_model

    """

    def __init__(
        self,
        J: Union[torch.Tensor, ndarray],
        h: Union[torch.Tensor, ndarray, None] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
        digital_ising_size: Optional[int] = 8,
        use_fpga: bool = False,
        weight_scale: int = 15
    ) -> None:
        self.dimension = J.shape[0]
        if isinstance(J, ndarray):
            J = torch.from_numpy(J)
        if isinstance(h, ndarray):
            h = torch.from_numpy(h)
        self.__init_from_tensor(J, h, dtype, device)
        self.computed_spins = None
        self.digital_ising_size = digital_ising_size
        self.weight_scale = weight_scale
        self.use_fpga = use_fpga
        self.time_elapsed = None
        if use_fpga:
            self.ising_lib = ctypes.CDLL("/usr/lib64/ising_lib.so")
            self.ising_lib.initialize_fpga()

    def __len__(self) -> int:
        return self.dimension

    def __neg__(self) -> SelfIsing:
        return self.__class__(-self.J, -self.h, self.dtype, self.device)

    def __init_from_tensor(
        self,
        J: torch.Tensor,
        h: Optional[torch.Tensor],
        dtype: torch.dtype,
        device: Union[str, torch.device],
    ) -> None:
        null_vector = torch.zeros(self.dimension, dtype=dtype, device=device)
        self.J = J.to(device=device, dtype=dtype)
        if h is None:
            self.h = null_vector
            self.linear_term = False
        else:
            self.h = h.to(device=device, dtype=dtype)
            self.linear_term = not torch.equal(self.h, null_vector)

    def clip_vector_to_tensor(self) -> torch.Tensor:
        """
        Gather `self.J` and `self.h` into a single matrix.

        The output matrix describes an equivalent Ising model in dimension
        `self.dimension + 1` with no linear term.

        Returns
        -------
        tensor: Tensor
            Matrix describing the new Ising model.

        Notes
        -----
        The output matrix is defined as the following block matrix.

            (            |    )
            (   self.J   | -h )
            (____________|____)
            (    -h.T    |  0 )

        This matrix describes another Ising model `other` with no linear
        term in dimension `self.dimension + 1`, with the same minimal
        energy, and with a one to two correspondence between the ground
        states of the two models defined as follows.

        Ground states of `self` → Ground states of `other` ~ R^n x R

        s ↦ {(s, 1), (-s, -1)}

        """
        tensor = torch.zeros(
            (self.dimension + 1, self.dimension + 1),
            dtype=self.dtype,
            device=self.device,
        )
        tensor[: self.dimension, : self.dimension] = self.J
        tensor[: self.dimension, self.dimension] = -self.h
        tensor[self.dimension, : self.dimension] = -self.h
        return tensor

    @staticmethod
    def remove_diagonal_(tensor: torch.Tensor) -> None:
        """
        Fill the diagonal of `tensor` with zeros.

        Parameters
        ----------
        tensor : Tensor
            Tensor whose diagonal is filled with zeros.

        Returns
        -------
        None
            The input is modified in place.

        """
        torch.diagonal(tensor)[...] = 0

    @staticmethod
    def symmetrize(tensor: torch.Tensor) -> torch.Tensor:
        """
        Return the symmetric tensor defining the same quadratic form.

        Parameters
        ----------
        tensor : Tensor
            Tensor defining the quadratic form.

        Returns
        -------
        Tensor
            Symmetric tensor defining the same quadratic form as `tensor`.

        """
        return (tensor + tensor.t()) / 2.0

    def as_simulated_bifurcation_tensor(self) -> torch.Tensor:
        """
        Turn the instance into a tensor compatible with the SB algorithm.

        The SB algorithm runs on Ising models with no linear term, and
        whose matrix is symmetric and has only zeros on its diagonal.

        Returns
        -------
        sb_tensor : Tensor
            Equivalent tensor compatible with the SB algorithm.

        """
        tensor = self.symmetrize(self.J)
        self.remove_diagonal_(tensor)
        if self.linear_term:
            sb_tensor = self.clip_vector_to_tensor()
        else:
            sb_tensor = tensor
        return sb_tensor

    def program_weight(
        self,
        weight: int,
        addr: int,
        retries: int = 5,
        error : bool = True      # Throw an error if read value != written value.
                                 # Set to false if writing to a non-readable addr. 
    ) -> None:
        written = self.ising_lib.write_ising(weight, addr)
        if error:
            tries = 0
            if (written != weight) and (tries < retries):
                written = self.ising_lib.write_ising(weight, addr)
            assert(written == weight), "ERROR: Wrote " + hex(weight) + " to addr " +\
                                        hex(addr) + " but read " + hex(written)

    def program_digital_ising(
        self,
        order: List[int],
        autoscale: bool = False, # Scale weights to fit in the ising machine range.
                                 # If false, warn on weights that don't match.
        automerge: bool = True,  # For models smaller than 1/2 the solver size, merge
                                 # mutliple spins into multi-spin chunks.
        retries: int = 5,        # Number of times to retry a failed weight program 
        initial_spins: [ndarray, None] = None
    ) -> None:
        """
        Program the Digital Ising Machine using the provided Ising
        tensor.

        Digital Ising Machine contains digital_ising_size physical
        spins. The final spin is the local field potential.
        """
        self.remove_diagonal_(self.J)
        J_list = self.J.numpy() # Don't symmetrize, we support asymmetric coupling
        h_list = self.h.numpy()

        if automerge:
            mult   = int(self.digital_ising_size/J_list.shape[0])
            J_list = np.kron(J_list, np.ones((mult,mult)))
            h_list = np.kron(h_list, np.ones(mult))
            if initial_spins is not None:
                initial_spins = np.kron(initial_spins, np.ones(mult))
       
        if autoscale:
            if self.linear_term:
                max_val = max((np.max(np.absolute(J_list)), np.max(np.absolute(h_list))))
            else:
                max_val = np.max(np.absolute(J_list))
            scale = int(self.weight_scale/2) / max_val
            J_list *= scale
            h_list *= scale

        default_weight = int(self.weight_scale/2)
        valid_weights  = range(-int(self.weight_scale/2), int(self.weight_scale/2) + 1)

        for i in range(0, self.digital_ising_size - 1):
            for j in range(0, self.digital_ising_size - 1):
                if (i != j):
                    if (i < J_list.shape[0]) and (j < J_list.shape[1]):
                        weight_val = J_list[i][j]
                        if weight_val not in valid_weights:
                            warnings.warn("Rounding weight "+str(i)+","+str(j)+". Was "+str(weight_val))
                            if weight_val >  int(self.weight_scale/2) : weight_val =  int(self.weight_scale/2)
                            if weight_val < -int(self.weight_scale/2) : weight_val = -int(self.weight_scale/2)
                        weight = int(int(weight_val) + int(self.weight_scale/2))
                    else:
                        weight = default_weight
                    addr = 0x01000000 + (order[j] << 13) + (order[i] << 2)
                    self.program_weight(weight, addr, retries = retries, error = True)
                else:
                    spin = 1 if (initial_spins is None or initial_spins[i] == 1) else 0
                    addr = 0x01000000 + (order[j] << 13) + (order[i] << 2)
                    self.program_weight(spin, addr, retries = retries, error = False) #TODO: can't read spins

            if self.linear_term:
                if (i < h_list.shape[0]):
                    weight_val = h_list[i]
                    if weight_val not in valid_weights:
                        warnings.warn("Rounding local field weight "+str(i)+". Was "+str(weight_val))
                        if weight_val >  int(self.weight_scale/2) : weight_val =  int(self.weight_scale/2)
                        if weight_val < -int(self.weight_scale/2) : weight_val = -int(self.weight_scale/2)
                    weight = int(int(self.weight_scale/2) - int(weight_val))
                else:
                    weight = default_weight
                # TODO: How to represent an asymmetric H?
                addr = 0x01000000 + ((self.digital_ising_size - 1)<<13) + (order[i] << 2 );
                self.program_weight(weight, addr)
                addr = 0x01000000 + ((self.digital_ising_size - 1)<<2 ) + (order[i] << 13);
                self.program_weight(weight, addr)
                # TODO: initial H value is not programmed

        return order

    def configure_digital_ising(
        self,
        counter_cutoff: int = 0x00004000,
        counter_max: int = 0x00008000
    ) -> None:
        """
        Configure the counters on the digital ising machine.
        
        Parameters
        ----------
        counter_cutoff : int
            The phase counter value at which a spin is considered "in phase"
            with the local field potential.
        counter_max : int
            The phase counter value at which the counter overflows and stops
            counting up. Usually 2x counter_cutoff.
        """
        self.ising_lib.write_ising(counter_cutoff, 0x00000600)
        self.ising_lib.write_ising(counter_max   , 0x00000700)

    def run_digital_ising(
        self,
        order: List[int],
        agents: int = 1,
        counter_cutoff: int = 0x00004000,
        cycles: int = 1000,
        automerge: bool = True
    ) -> List[int]:
        """
        Run the digital Ising machine!
        
        Parameters
        ----------
        agents  : int
            Number of times to run the solver.
            TODO: Parallelism is not supported right now.
        counter_cutoff : int 
            The phase counter value at which a spin is considered "in phase"
            with the local field potential.
        time_ms : int
            The time, in milliseconds, to wait before reading out data.
        """
        mult = 1
        if automerge:
            mult = int(self.digital_ising_size/len(self.J))
        
        spins = [[] for _ in range(len(self.J))]
        for j in range(agents):
            start = time.time()
            self.ising_lib.write_ising(int(cycles), 0x00000500) # Start
            time.sleep(0.0001)
            finish = time.time()
            self.time_elapsed += finish - start
            for i in range(len(self.J) * mult):
                if (i % mult == 0):
                    merged = np.zeros(mult)

                index = order[i]
                addr = 0x00001000 + (index << 2)
                value = self.ising_lib.read_ising(addr)
                spin = 1 if (value > counter_cutoff) else -1
                merged[i % mult] = spin

                if (i % mult == (mult-1)): 
                    if(merged != merged[0]).all():
                        warnings.warn("Merged spins don't match for elem "+str(i))
                    spins[int(i / mult)].append(merged[0])

        return spins

    def get_energy(self) -> int:
        if self.computed_spins is None:
            raise Exception("Can't get energy if spins aren't computed!")

        if self.linear_term:
            return -0.5*torch.matmul(torch.matmul(torch.transpose(self.computed_spins,0,1),self.J),self.computed_spins)[0]
        else:
            return torch.matmul(torch.transpose(self.h,0,1),self.computed_spins)[0] - \
                   0.5*torch.matmul(torch.matmul(torch.transpose(self.computed_spins,0,1),self.J),self.computed_spins)[0]

    def get_rand_energy(self) -> int:
        random_spins = torch.randint(0, 2, (len(self.J),1), dtype=torch.float32)
        random_spins = torch.add(torch.mul(random_spins, 2), -1)

        if self.linear_term:
            return -0.5*torch.matmul(torch.matmul(torch.transpose(random_spins,0,1),self.J),random_spins)[0]
        else:
            return torch.matmul(torch.transpose(self.h,0,1),self.computed_spins)[0] - \
                   0.5*torch.matmul(torch.matmul(torch.transpose(random_spins,0,1),self.J),random_spins)[0]

    @property
    def dtype(self) -> torch.dtype:
        """
        torch.dtype:
            Data-type of the coefficients of the Ising model.

        """
        return self.J.dtype

    @property
    def device(self) -> torch.device:
        """
        torch.device:
            Device on which the Ising model is located.

        """
        return self.J.device

    def minimize(
        self,
        agents: int = 128,
        max_steps: int = 10000,
        ballistic: bool = False,
        heated: bool = False,
        verbose: bool = True,
        *,
        use_window: bool = True,
        sampling_period: int = 50,
        convergence_threshold: int = 50,
        timeout: Optional[float] = None,
        use_fpga: bool = False,
        autoscale: bool = True,
        automerge: bool = True,
        counter_cutoff: int = 0x00004000,
        counter_max: int = 0x00008000,
        cycles: int = 1000,
        shuffle_spins: bool = False,
        weight_program_retries: int = 5,
        initial_spins: [ndarray, None] = None
    ) -> None:
        """
        Minimize the energy of the Ising model using the Simulated Bifurcation
        algorithm.

        Parameters
        ----------
        agents : int, default=128
            Number of simultaneous execution of the SB algorithm. This is
            much faster than sequentially running the SB algorithm `agents`
            times.
        max_steps : int, default=10_000
            Number of iterations after which the algorithm is stopped
            regardless of whether convergence has been achieved.
        ballistic : bool, default=False
            Whether to use the ballistic or the discrete SB algorithm.
            See Notes for further information about the variants of the SB
            algorithm.
        heated : bool, default=False
            Whether to use the heated or non-heated SB algorithm.
            See Notes for further information about the variants of the SB
            algorithm.
        verbose : bool, default=True
            Whether to display a progress bar to monitor the progress of
            the algorithm.
        use_window : bool, default=True
            Whether to use the window as a stopping criterion: an agent is
            said to have converged if its energy has not changed over the
            last `convergence_threshold` energy samplings (done every
            `sampling_period` steps).
        sampling_period : int, default=50
            Number of iterations between two consecutive energy samplings
            by the window.
        convergence_threshold : int, default=50
            Number of consecutive identical energy samplings considered as
            a proof of convergence by the window.
        timeout : float | None, default=None
            Time in seconds after which the simulation is stopped.
            None means no timeout.

        Returns
        -------
        None
            The spins of all agents returned by the SB algorithm are stored
            in the `computed_spins` attribute.

        Other Parameters
        ----------------
        Hyperparameters corresponding to physical constants :
            These parameters have been fine-tuned (Goto et al.) to give the
            best results most of the time. Nevertheless, the relevance of
            specific hyperparameters may vary depending on the properties
            of the instances. They can respectively be modified and reset
            through the `set_env` and `reset_env` functions.

        Warns
        -----
        If `use_window` is True and no agent has reached the convergence
        criterion defined by `sampling_period` and `convergence_threshold`
        within `max_steps` iterations, a warning is logged in the console.
        This is just an indication however; the returned vectors may still
        be of good quality. Solutions to this warning include:

        - increasing the time step in the SB algorithm (may decrease
          numerical stability), see the `set_env` function.
        - increasing `max_steps` (at the expense of runtime).
        - changing the values of `ballistic` and `heated` to use
          different variants of the SB algorithm.
        - changing the values of some hyperparameters corresponding to
          physical constants (advanced usage, see Other Parameters).

        Warnings
        --------
        Approximation algorithm:
            The SB algorithm is an approximation algorithm, which implies
            that the returned values may not correspond to global optima.
            Therefore, if some constraints are embedded as penalties in the
            polynomial, that is adding terms that ensure that any global
            optimum satisfies the constraints, the return values may
            violate these constraints.
        Non-deterministic behaviour:
            The SB algorithm uses a randomized initialization, and this
            package is implemented with a PyTorch backend. To ensure a
            consistent initialization when running the same script multiple
            times, use `torch.manual_seed`. However, results may not be
            reproducible between CPU and GPU executions, even when using
            identical seeds. Furthermore, certain PyTorch operations are
            not deterministic. For more comprehensive details on
            reproducibility, refer to the PyTorch documentation available
            at https://pytorch.org/docs/stable/notes/randomness.html.

        See Also
        --------
        models.Ising:
            Implementation of the Ising model which behaves like other
            models and polynomials.
        QuadraticPolynomial:
            Class to implement multivariate quadratic polynomials from SymPy
            polynomial expressions or tensors that can be casted to Ising model
            for Simulated Bifurcation algorithm compatibility purposes.

        Notes
        -----
        The original version of the SB algorithm [1] is not implemented
        since it is less efficient than the more recent variants of the SB
        algorithm described in [2]:

        - ballistic SB : Uses the position of the particles for the
          position-based update of the momentums ; usually faster but
          less accurate. Use this variant by setting
          `ballistic=True`.
        - discrete SB : Uses the sign of the position of the particles
          for the position-based update of the momentums ; usually
          slower but more accurate. Use this variant by setting
          `ballistic=False`.

        On top of these two variants, an additional thermal fluctuation
        term can be added in order to help escape local optima [3]. Use
        this additional term by setting `heated=True`.

        The space complexity O(M^2 + `agents` * M). The time complexity is
        O(`max_steps` * `agents` * M^2) where M is the dimension of the
        instance.

        For instances in low dimension (~100), running computations on GPU
        is slower than running computations on CPU unless a large number of
        agents (~2000) is used.

        References
        ----------
        [1] Hayato Goto et al., "Combinatorial optimization by simulating
        adiabatic bifurcations in nonlinear Hamiltonian systems". Sci.
        Adv.5, eaav2372(2019). DOI:10.1126/sciadv.aav2372
        [2] Hayato Goto et al., "High-performance combinatorial
        optimization based on classical mechanics". Sci. Adv.7,
        eabe7953(2021). DOI:10.1126/sciadv.abe7953
        [3] Kanao, T., Goto, H. "Simulated bifurcation assisted by thermal
        fluctuation". Commun Phys 5, 153 (2022).
        https://doi.org/10.1038/s42005-022-00929-9

        """
        if use_fpga:
            if self.linear_term: order = np.arange(self.digital_ising_size - 1) 
            else               : order = np.arange(self.digital_ising_size    ) 
            if shuffle_spins: np.random.shuffle(order)
            order = order.tolist() # for typing reasons
            order = [int(_) for _ in order] # for typing reasons
            self.time_elapsed = 0
            self.configure_digital_ising(
                 counter_cutoff = counter_cutoff,
                 counter_max = counter_max
            )
            order = self.program_digital_ising(
                 autoscale = autoscale,
                 automerge = automerge,
                 order = order,
                 retries = weight_program_retries,
                 initial_spins = initial_spins
            )
            spins = self.run_digital_ising(
                 agents = agents,
                 counter_cutoff = counter_cutoff,
                 cycles = cycles,
                 automerge = automerge,
                 order = order
            )
            self.computed_spins = torch.Tensor(spins)
        else:
            engine = SimulatedBifurcationEngine.get_engine(ballistic, heated)
            optimizer = SimulatedBifurcationOptimizer(
                agents,
                max_steps,
                timeout,
                engine,
                verbose,
                sampling_period,
                convergence_threshold,
            )
            tensor = self.as_simulated_bifurcation_tensor()
            start = time.time()
            spins = optimizer.run_integrator(tensor, use_window)
            finish = time.time()
            self.time_elapsed = finish - start
            if self.linear_term:
                self.computed_spins = spins[-1] * spins[:-1]
            else:
                self.computed_spins = spins
