import time
import os


class UpdateProgress():
    def __init__(self, total_iters: int) -> None:
        """
        A class to give updates about the status for long running tasks.
        Should initally be called with the total number of iterations the task
        is to be run for and then called regurlarly with update_status.
        """
        self.total_iters = total_iters
        self.start_time = time.time()

    def __clear__(self) -> None:
        """Clears the terminal window"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def __get_time_string__(self, time_in_secs: int) -> str:
        """
        Transforms a time in seconds as an integer into a human readable
        time-string
        """
        hour_time = int(time_in_secs // 3600)
        min_time = int(time_in_secs // 60) % 60
        sec_time = int(time_in_secs % 60)
        if hour_time:
            return f"{hour_time} h {min_time} min {sec_time} sec"
        elif min_time:
            return f"{min_time} min {sec_time} sec"
        else:
            return f"{sec_time} sec"

    def __remaining_time__(self, num_iter: int, elapsed_time: int) -> str:
        """
        Calculates the estimated remaining time given how far along the process
        is right now and how long it has taken until now.
        """
        if num_iter == 0:
            return "Calculating..."
        # Simplification of the formula mult = (1.0 / frac_done) - 1.0, where
        # frac_done = num_iter / self.total_iters
        multiplier = (float(self.total_iters) / num_iter) - 1.0
        remaining_time_in_secs = multiplier * elapsed_time
        return self.__get_time_string__(remaining_time_in_secs)

    def reset_start_time(self) -> None:
        """
        Method to reset the start time of the updater to current time.
        """
        self.start_time = time.time()

    def update_status(self, num_iter: int, msg: str = None) -> None:
        """
        Clears the windows and updates the status of the process. Should be
        called with the current iteration number. If needed, it can also be
        supplied an extra message that will be printed above the other info.
        NB: Should probably not be called on every iteration as all the
        printing will be slow so do something like

        >>> if num_iter % 500 == 0:
        >>>     updater.update_status(num_iter)

        where you use a fitting number to do modulo with
        """
        self.__clear__()
        if msg:
            print(msg)

        elapsed_time = time.time() - self.start_time
        percent_done = 100 * (num_iter / float(self.total_iters))
        elaped_time_str = self.__get_time_string__(elapsed_time)
        rem_time_str = self.__remaining_time__(num_iter, elapsed_time)
        print(f"Iteration:      {num_iter} / {self.total_iters} " +
              f"({percent_done:3.2f} %)")
        print(f"Elapsed time:   {elaped_time_str}")
        print(f"Remaining time: {rem_time_str}")
