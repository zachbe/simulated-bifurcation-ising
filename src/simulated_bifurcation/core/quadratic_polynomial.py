import re
from typing import Optional, Tuple, Union

import numpy as np
import torch

from ..polynomial.polynomial import Polynomial, PolynomialLike
from .ising import Ising

INTEGER_REGEX = re.compile("^int[1-9][0-9]*$")


class QuadraticPolynomialError(ValueError):
    def __init__(self, degree: int) -> None:
        super().__init__(f"Expected a degree 2 polynomial, got {degree}.")


class QuadraticPolynomial(Polynomial):
    def __init__(
        self,
        *polynomial_like: PolynomialLike,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__(*polynomial_like, dtype=dtype, device=device)
        if self.degree != 2:
            raise QuadraticPolynomialError(self.degree)

    def __call__(self, value: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            try:
                value = torch.tensor(value, dtype=self.dtype, device=self.device)
            except Exception as err:
                raise TypeError("Input value cannot be cast to Tensor.") from err

        if value.shape[-1] != self.dimension:
            raise ValueError(
                f"Size of the input along the last axis should be "
                f"{self.dimension}, it is {value.shape[-1]}."
            )

        quadratic_term = torch.nn.functional.bilinear(
            value,
            value,
            torch.unsqueeze(self[2], 0),
        )
        affine_term = value @ self[1] + self[0]
        evaluation = torch.squeeze(quadratic_term, -1) + affine_term
        return evaluation

    def to_ising(self, input_type: str) -> Ising:
        """
        Generate an equivalent Ising model of the problem.
        The notion of equivalence means that finding the ground
        state of this new model is strictly equivalent to find
        the ground state of the original problem.

        Returns
        -------
        IsingCore
        """
        if input_type == "spin":
            return Ising(-2 * self[2], self[1], self.dtype, self.device)
        if input_type == "binary":
            symmetrical_matrix = Ising.symmetrize(self[2])
            J = -0.5 * symmetrical_matrix
            h = 0.5 * self[1] + 0.5 * symmetrical_matrix @ torch.ones(
                len(self), dtype=self.dtype, device=self.device
            )
            return Ising(J, h, self.dtype, self.device)
        if INTEGER_REGEX.match(input_type) is None:
            raise ValueError(
                f'Input type must be one of "spin" or "binary", or be a string starting'
                f'with "int" and be followed by a positive integer.\n'
                f"More formally, it should match the following regular expression.\n"
                f"{INTEGER_REGEX}\n"
                f'Examples: "int7", "int42", ...'
            )
        number_of_bits = int(input_type[3:])
        symmetrical_matrix = Ising.symmetrize(self[2])
        integer_to_binary_matrix = QuadraticPolynomial.integer_to_binary_matrix(
            self.dimension, number_of_bits, device=self.device
        )
        J = (
            -0.5
            * integer_to_binary_matrix
            @ symmetrical_matrix
            @ integer_to_binary_matrix.t()
        )
        h = 0.5 * integer_to_binary_matrix @ self[
            1
        ] + 0.5 * integer_to_binary_matrix @ self[
            2
        ] @ integer_to_binary_matrix.t() @ torch.ones(
            (self.dimension * number_of_bits),
            dtype=self.dtype,
            device=self.device,
        )
        return Ising(J, h, self.dtype, self.device)

    # TODO: add decorator @type_parser(if_spin, if_bin, if_int, bits)

    def convert_spins(self, ising: Ising, input_type: str) -> Optional[torch.Tensor]:
        """
        Retrieves information from the optimized equivalent Ising model.
        Returns the best found vector if `ising.ground_state` is not `None`.
        Returns `None` otherwise.

        Parameters
        ----------
        ising : IsingCore
            Equivalent Ising model of the problem.

        Returns
        -------
        Tensor
        """
        if ising.computed_spins is not None:
            return None
        if input_type == "spin":
            return ising.computed_spins
        if input_type == "binary":
            return (ising.computed_spins + 1) / 2
        if INTEGER_REGEX.match(input_type) is None:
            raise ValueError(
                f'Input type must be one of "spin" or "binary", or be a string starting'
                f'with "int" and be followed by a positive integer.\n'
                f"More formally, it should match the following regular expression.\n"
                f"{INTEGER_REGEX}\n"
                f'Examples: "int7", "int42", ...'
            )
        number_of_bits = int(input_type[3:])
        integer_to_binary_matrix = QuadraticPolynomial.integer_to_binary_matrix(
            self.dimension, number_of_bits, device=self.device
        )
        return 0.5 * integer_to_binary_matrix.t() @ (ising.computed_spins + 1)

    def optimize(
        self,
        input_type: str,
        agents: int = 128,
        max_steps: int = 10000,
        best_only: bool = True,
        ballistic: bool = False,
        heated: bool = False,
        minimize: bool = True,
        verbose: bool = True,
        *,
        use_window: bool = True,
        sampling_period: int = 50,
        convergence_threshold: int = 50,
        timeout: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes a local extremum of the model by optimizing
        the equivalent Ising model using the Simulated Bifurcation (SB)
        algorithm.

        The Simulated Bifurcation (SB) algorithm relies on
        Hamiltonian/quantum mechanics to find local minima of
        Ising problems. The spins dynamics is simulated using
        a first order symplectic integrator.

        There are different version of the SB algorithm:
        - the ballistic Simulated Bifurcation (bSB) which uses the particles'
        position for the matrix computations (usually faster but less accurate)
        - the discrete Simulated Bifurcation (dSB) which uses the particles'
        spin for the matrix computations (usually slower but more accurate)
        - the Heated ballistic Simulated Bifurcation (HbSB) which uses the bSB
        algorithm with a supplementary non-symplectic term to refine the model
        - the Heated ballistic Simulated Bifurcation (HdSB) which uses the dSB
        algorithm with a supplementary non-symplectic term to refine the model

        To stop the iterations of the symplectic integrator, a number of maximum
        steps needs to be specified. However, a refined way to stop is also possible
        using a window that checks that the spins have not changed among a set
        number of previous steps. In practice, a every fixed number of steps
        (called a sampling period) the current spins will be compared to the
        previous ones. If they remain constant throughout a certain number of
        consecutive samplings (called the convergence threshold), the spins are
        considered to have bifurcated and the algorithm stops.

        Finally, it is possible to make several particle vectors at the same
        time (each one being called an agent). As the vectors are randomly
        initialized, using several agents helps to explore the solution space
        and increases the probability of finding a better solution, though it
        also slightly increases the computation time. In the end, only the best
        spin vector (energy-wise) is kept and used as the new Ising model's
        ground state.

        Parameters
        ----------
        *
        convergence_threshold : int, optional
            number of consecutive identical spin sampling considered as a proof
            of convergence (default is 50)
        sampling_period : int, optional
            number of time steps between two spin sampling (default is 50)
        max_steps : int, optional
            number of time steps after which the algorithm will stop inevitably
            (default is 10000)
        agents : int, optional
            number of vectors to make evolve at the same time (default is 128)
        use_window : bool, optional
            indicates whether to use the window as a stopping criterion or not
            (default is True)
        timeout : float | None, default=None
            Time in seconds after which the simulation is stopped.
            None means no timeout.
        ballistic : bool, optional
            if True, the ballistic SB will be used, else it will be the
            discrete SB (default is True)
        heated : bool, optional
            if True, the heated SB will be used, else it will be the non-heated
            SB (default is True)
        verbose : bool, optional
            whether to display a progress bar to monitor the algorithm's
            evolution (default is True)
        best_only : bool, optional
            if `True` only the best found solution to the optimization problem
            is returned, otherwise all the solutions found by the simulated
            bifurcation algorithm.
        minimize : bool, optional
            if `True` the optimization direction is minimization, otherwise it
            is maximization (default is True)

        Returns
        -------
        Tensor
        """
        if minimize:
            ising_equivalent = self.to_ising(input_type)
        else:
            ising_equivalent = -self.to_ising(input_type)
        ising_equivalent.minimize(
            agents,
            max_steps,
            ballistic,
            heated,
            verbose,
            use_window=use_window,
            sampling_period=sampling_period,
            convergence_threshold=convergence_threshold,
            timeout=timeout,
        )
        self.sb_result = self.convert_spins(ising_equivalent, input_type)
        result = self.sb_result.t()
        evaluation = self(result)
        if best_only:
            i_best = torch.argmin(evaluation) if minimize else torch.argmax(evaluation)
            result = result[i_best]
            evaluation = evaluation[i_best]
        return result, evaluation

    def minimize(
        self,
        input_type: str,
        agents: int = 128,
        max_steps: int = 10000,
        best_only: bool = True,
        ballistic: bool = False,
        heated: bool = False,
        verbose: bool = True,
        *,
        use_window: bool = True,
        sampling_period: int = 50,
        convergence_threshold: int = 50,
        timeout: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes a local minimum of the model by optimizing
        the equivalent Ising model using the Simulated Bifurcation (SB)
        algorithm.

        The Simulated Bifurcation (SB) algorithm relies on
        Hamiltonian/quantum mechanics to find local minima of
        Ising problems. The spins dynamics is simulated using
        a first order symplectic integrator.

        There are different version of the SB algorithm:
        - the ballistic Simulated Bifurcation (bSB) which uses the particles'
        position for the matrix computations (usually faster but less accurate)
        - the discrete Simulated Bifurcation (dSB) which uses the particles'
        spin for the matrix computations (usually slower but more accurate)
        - the Heated ballistic Simulated Bifurcation (HbSB) which uses the bSB
        algorithm with a supplementary non-symplectic term to refine the model
        - the Heated ballistic Simulated Bifurcation (HdSB) which uses the dSB
        algorithm with a supplementary non-symplectic term to refine the model

        To stop the iterations of the symplectic integrator, a number of maximum
        steps needs to be specified. However, a refined way to stop is also possible
        using a window that checks that the spins have not changed among a set
        number of previous steps. In practice, a every fixed number of steps
        (called a sampling period) the current spins will be compared to the
        previous ones. If they remain constant throughout a certain number of
        consecutive samplings (called the convergence threshold), the spins are
        considered to have bifurcated and the algorithm stops.

        Finally, it is possible to make several particle vectors at the same
        time (each one being called an agent). As the vectors are randomly
        initialized, using several agents helps to explore the solution space
        and increases the probability of finding a better solution, though it
        also slightly increases the computation time. In the end, only the best
        spin vector (energy-wise) is kept and used as the new Ising model's
        ground state.

        Parameters
        ----------
        *
        convergence_threshold : int, optional
            number of consecutive identical spin sampling considered as a proof
            of convergence (default is 50)
        sampling_period : int, optional
            number of time steps between two spin sampling (default is 50)
        max_steps : int, optional
            number of time steps after which the algorithm will stop inevitably
            (default is 10000)
        agents : int, optional
            number of vectors to make evolve at the same time (default is 128)
        use_window : bool, optional
            indicates whether to use the window as a stopping criterion or not
            (default is True)
        timeout : float | None, default=None
            Time in seconds after which the simulation is stopped.
            None means no timeout.
        ballistic : bool, optional
            if True, the ballistic SB will be used, else it will be the
            discrete SB (default is True)
        heated : bool, optional
            if True, the heated SB will be used, else it will be the non-heated
            SB (default is True)
        verbose : bool, optional
            whether to display a progress bar to monitor the algorithm's
            evolution (default is True)
        best_only : bool, optional
            if `True` only the best found solution to the optimization problem
            is returned, otherwise all the solutions found by the simulated
            bifurcation algorithm.

        Returns
        -------
        Tensor
        """
        return self.optimize(
            input_type,
            agents,
            max_steps,
            best_only,
            ballistic,
            heated,
            True,
            verbose,
            use_window=use_window,
            sampling_period=sampling_period,
            convergence_threshold=convergence_threshold,
            timeout=timeout,
        )

    def maximize(
        self,
        input_type: str,
        agents: int = 128,
        max_steps: int = 10000,
        best_only: bool = True,
        ballistic: bool = False,
        heated: bool = False,
        verbose: bool = True,
        *,
        use_window: bool = True,
        sampling_period: int = 50,
        convergence_threshold: int = 50,
        timeout: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes a local maximum of the model by optimizing
        the equivalent Ising model using the Simulated Bifurcation (SB)
        algorithm.

        The Simulated Bifurcation (SB) algorithm relies on
        Hamiltonian/quantum mechanics to find local minima of
        Ising problems. The spins dynamics is simulated using
        a first order symplectic integrator.

        There are different version of the SB algorithm:
        - the ballistic Simulated Bifurcation (bSB) which uses the particles'
        position for the matrix computations (usually faster but less accurate)
        - the discrete Simulated Bifurcation (dSB) which uses the particles'
        spin for the matrix computations (usually slower but more accurate)
        - the Heated ballistic Simulated Bifurcation (HbSB) which uses the bSB
        algorithm with a supplementary non-symplectic term to refine the model
        - the Heated ballistic Simulated Bifurcation (HdSB) which uses the dSB
        algorithm with a supplementary non-symplectic term to refine the model

        To stop the iterations of the symplectic integrator, a number of maximum
        steps needs to be specified. However, a refined way to stop is also possible
        using a window that checks that the spins have not changed among a set
        number of previous steps. In practice, a every fixed number of steps
        (called a sampling period) the current spins will be compared to the
        previous ones. If they remain constant throughout a certain number of
        consecutive samplings (called the convergence threshold), the spins are
        considered to have bifurcated and the algorithm stops.

        Finally, it is possible to make several particle vectors at the same
        time (each one being called an agent). As the vectors are randomly
        initialized, using several agents helps to explore the solution space
        and increases the probability of finding a better solution, though it
        also slightly increases the computation time. In the end, only the best
        spin vector (energy-wise) is kept and used as the new Ising model's
        ground state.

        Parameters
        ----------
        convergence_threshold : int, optional
            number of consecutive identical spin sampling considered as a proof
            of convergence (default is 50)
        sampling_period : int, optional
            number of time steps between two spin sampling (default is 50)
        max_steps : int, optional
            number of time steps after which the algorithm will stop inevitably
            (default is 10000)
        agents : int, optional
            number of vectors to make evolve at the same time (default is 128)
        use_window : bool, optional
            indicates whether to use the window as a stopping criterion or not
            (default is True)
        timeout : float | None, default=None
            Time in seconds after which the simulation is stopped.
            None means no timeout.
        ballistic : bool, optional
            if True, the ballistic SB will be used, else it will be the
            discrete SB (default is True)
        heated : bool, optional
            if True, the heated SB will be used, else it will be the non-heated
            SB (default is True)
        verbose : bool, optional
            whether to display a progress bar to monitor the algorithm's
            evolution (default is True)
        best_only : bool, optional
            if `True` only the best found solution to the optimization problem
            is returned, otherwise all the solutions found by the simulated
            bifurcation algorithm.

        Returns
        -------
        Tensor
        """
        return self.optimize(
            input_type,
            agents,
            max_steps,
            best_only,
            ballistic,
            heated,
            False,
            verbose,
            use_window=use_window,
            sampling_period=sampling_period,
            convergence_threshold=convergence_threshold,
            timeout=timeout,
        )

    @staticmethod
    def integer_to_binary_matrix(
        dimension: int, number_of_bits: int, device: Union[str, torch.device]
    ) -> torch.Tensor:
        matrix = torch.zeros((dimension * number_of_bits, dimension), device=device)
        for row in range(dimension):
            for col in range(number_of_bits):
                matrix[row * number_of_bits + col][row] = 2.0**col
        return matrix
