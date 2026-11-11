from dataclasses import dataclass
from warnings import warn

import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageChops

@dataclass
class ClosureODEParams:
    nu_zero: float
    a: float
    tau: float = None
    noise_sigma: float = None
    gamma_star: float = None
    ring_tension: float = None
    diffusivity: float = None

NOISE_TYPES = ['covariance_matrix', 'sliding_window']
NOISE_SAMPLING_TYPES = ['tau', 'always']

class ClosureODE:
    def __init__(
            self,
            params: ClosureODEParams,
            n_s: int,
            r_zero: float = 1.0,
            noise_type: str = 'sliding_window',
            noise_sampling = 'tau'
        ):
        self.params: ClosureODEParams = params
        self.r_zero = r_zero
        self.n_s = n_s
        self.smax = 2 * np.pi * self.r_zero

        if (self.params.gamma_star is None or self.params.ring_tension is None) and self.params.diffusivity is None:
            raise ValueError("Either (gamma_star and ring_tension) or diffusivity must be specified")
        if (self.params.gamma_star is not None or self.params.ring_tension is not None) and self.params.diffusivity is not None:
            warn("Both (gamma_star or ring_tension) and diffusivity are specified. Using diffusivity.")
        if self.params.diffusivity is None:
            self.params.diffusivity = (self.params.nu_zero * self.params.ring_tension) / self.params.gamma_star

        if self.params.noise_sigma is None:
            self.params.noise_sigma = self.params.nu_zero**2

        # check for valid noise type
        if noise_type not in NOISE_TYPES:
            raise ValueError(f"Unknown noise type: {noise_type}, select from {NOISE_TYPES}")
        self.noise_type = noise_type

        # noise recalculation function and rate kept separately in case of user change
        self.noise_function: callable[[float, float, np.ndarray], np.ndarray] = None
        if self.noise_type == 'covariance_matrix':
            self.noise_function = self.covariance_matrix_noise
        elif self.noise_type == 'sliding_window':
            self.noise_function = self.sliding_window_noise

        # check for valid sampling type and tau if needed
        if noise_sampling not in NOISE_SAMPLING_TYPES:
            raise ValueError(f"Unknown noise sampling type: {noise_sampling}, select from {NOISE_SAMPLING_TYPES}")
        self.noise_sampling = noise_sampling

        if self.noise_sampling == 'tau' and self.params.tau is None:
            raise ValueError("tau must be specified for sliding_window noise")
        self.t_last_noise: int = None
        
        self.s_grid = np.linspace(0, 1, self.n_s, endpoint=False)
        self.ds_grid = np.roll(self.s_grid, -1) - self.s_grid
        self.ds_grid[-1] += 1.0

        self.rng = np.random.default_rng()
        self.s_shaped_zeros = np.zeros_like(self.s_grid)

        # Buffers to save memory allocation and make shape checks automatic
        self.covariance_matrix = np.zeros((self.n_s, self.n_s))
        self.covariance_row = np.zeros(self.n_s)
        self.covariance_filler = np.zeros((self.n_s, self.n_s))
        self.eta = np.zeros_like(self.s_grid)
        self.d2h_ds2 = np.zeros_like(self.s_grid)
        self.ds = self.ds_grid * self.smax

    @staticmethod
    def xy_from_rh(rmean_per_t, h_per_s_per_t, s_grid):
        r_per_s_per_t = rmean_per_t + h_per_s_per_t

        xy_per_t_per_s = np.zeros((len(s_grid), 2, rmean_per_t.shape[-1]))
        xy_per_t_per_s[:, 0, :] = r_per_s_per_t * np.cos(s_grid*2*np.pi)[:, np.newaxis]
        xy_per_t_per_s[:, 1, :] = r_per_s_per_t * np.sin(s_grid*2*np.pi)[:, np.newaxis]

        return xy_per_t_per_s

    @staticmethod
    def calculate_d2h_ds2(h, ds, output: np.ndarray):
        # append and prepend here since we are working with periodic boundaries
        # do one append and one prepend cause otherwise gradient is shifted by one
        output[:] = np.diff(h, 2, prepend=h[-1], append=h[0])
        output /= ds**2

    def recalculate_noise(self, t):
        if self.noise_sampling == 'always':
            return True
        if self.noise_sampling == 'tau':
            return ((t - self.t_last_noise) >= self.params.tau)

    def dh_dt(self, t, current_h_of_s, mean_radius, output=None):
        if output is None:
            output = np.zeros_like(self.params.diffusivity, current_h_of_s)

        self.ds[:] = self.ds_grid * (2 * np.pi * mean_radius)
        
        if self.recalculate_noise(t):
            # in sent code this is only done once per tau seconds
            self.noise_function(mean_radius=mean_radius, t=t, output=self.eta)
            self.t_last_noise = t

        self.calculate_d2h_ds2(current_h_of_s, self.ds, output=self.d2h_ds2)

        output[:] = self.params.diffusivity
        output *= self.d2h_ds2
        output += self.eta

        return output

    def dmean_radius_dt(self, t, mean_radius):
        return -(self.params.nu_zero + (self.params.diffusivity / mean_radius))

    def vectorized_ode(self, t, y, output: np.ndarray=None):
        if output is None:
            output = np.zeros_like(y)

        # first element is the mean radius
        output[0] = self.dmean_radius_dt(t, y[0])
        # the rest are the h values with certain mean radius
        self.dh_dt(t=t, current_h_of_s=y[1:], mean_radius=y[0], output=output[1:])

        return output

    def calculate_covariance_matrix(self, dx, output: np.ndarray=None):
        if output is None:
            output = np.zeros((self.n_s, self.n_s))
        
        big_m = self.params.a / dx
        # print("M: ", big_m)
        n_corr = int(big_m)
        # print("correlated points: ", n_corr)
        factor = self.params.noise_sigma

        output[:, :] = 0.
        for corr_index in range(-n_corr, n_corr + 1):
            fill_value = 1. - (np.abs(corr_index) / big_m)
            # print(f"corr index: {corr_index}, fill value: {fill_value}")
            self.covariance_row[:] = fill_value*factor
            self.covariance_filler[:, :] = np.diag(self.covariance_row)
            output += np.roll(self.covariance_filler, corr_index, axis=0)

        return output

    def covariance_matrix_noise(self, t, mean_radius, output: np.ndarray=None):
        if output is None:
            output = np.zeros_like(self.s_shaped_zeros)
        
        dx = (2 * np.pi * mean_radius) / self.n_s
        self.calculate_covariance_matrix(dx, output=self.covariance_matrix)
        output[:] = self.rng.multivariate_normal(mean=self.s_shaped_zeros, cov=self.covariance_matrix)
        return output

    def sliding_window_noise(self, t, mean_radius, output: np.ndarray=None):
        if output is None:
            output = np.zeros_like(self.s_shaped_zeros)

        standard_normal = self.rng.normal(loc=0., scale=1., size=self.n_s)

        dx = (2 * np.pi * mean_radius) / self.n_s
        big_m = self.params.a / dx
        # no idea why this uses round instead of floor or ceil
        # also take into account cases with less than one correlated point, then at least use center
        n_corr = max(int(np.round(big_m)), 1)

        # apply nearest n_corr neighbors
        for ii_corr in range(0, n_corr-1):
            output += np.roll(standard_normal, ii_corr)
        output /= n_corr
        output *= np.sqrt(big_m*self.params.noise_sigma)

        return output
    
    def solve(self, stepsize, tmax, tmin=0., rmin=20.):
        t_values = np.arange(tmin, tmax, stepsize)
        # y contains mean radius in first element, then h values
        y_per_t = np.zeros((self.n_s + 1, len(t_values)))

        # initial condition
        y_per_t[0, 0] = self.r_zero

        # initialise noise
        self.t_last_noise = 0.
        self.noise_function(mean_radius=self.r_zero, t=0, output=self.eta)

        for ii_t, t_val in enumerate(t_values[1:]):
            # make gaussian step
            self.vectorized_ode(t_val, y_per_t[:, ii_t], output=y_per_t[:, ii_t+1])
            y_per_t[:, ii_t+1] *= stepsize
            y_per_t[:, ii_t+1] += y_per_t[:, ii_t]

            if np.min(y_per_t[1:, ii_t+1]+y_per_t[0, ii_t+1]) < rmin:
                # end loop if points are close enough to center
                break

        # return timesteps, mean radius, h values
        return t_values, y_per_t[0, :], y_per_t[1:, :]

class Plots:
    @staticmethod
    def plot_coordinate_frame(frame_coordinates, t_val, time_cmap, time_norm, ax:Axes):
            # add last element to close the loop
            draw_coordinates = np.append(frame_coordinates, frame_coordinates[0:1, :], axis=0)
            lines = ax.plot(
                draw_coordinates[:, 0], draw_coordinates[:, 1],
                color=time_cmap(time_norm(t_val))
            )
            return lines

    @staticmethod
    def plot_treerings(
        t_values, rmean_per_t, h_per_t, s_grid,
        fig:Figure=None, ax:Axes=None, time_cmap=None, time_norm=None,
        add_standard_labels=True
    ):
        coords_per_s_per_t = ClosureODE.xy_from_rh(rmean_per_t, h_per_t, s_grid)

        if (fig is None) and (ax is None):
            fig, ax = plt.subplots()
        elif fig is None:
            raise ValueError("Specify matplotlib.figure.Figure object in 'fig' when supplying matplotlib.axes.Axes object in 'ax'")
        elif ax is None:
            ax = fig.add_subplot(1, 1, 1)

        if time_cmap is None:
            time_cmap = plt.get_cmap('viridis')
        if time_norm is None:
            time_norm = plt.Normalize(vmin=0, vmax=np.max(t_values))

        for ii_t, t_val in enumerate(t_values):
            Plots.plot_coordinate_frame(coords_per_s_per_t[:, :, ii_t], t_val, time_cmap, time_norm, ax)

        cb = fig.colorbar(plt.cm.ScalarMappable(norm=time_norm, cmap=time_cmap), ax=ax)

        if add_standard_labels:
            cb.set_label('Time [s]')
            ax.set_xlabel('x [nm]')
            ax.set_ylabel('y [nm]')
            ax.set_title('Septal closure ODE solution')
            ax.set_aspect('equal')

        ax.set_aspect('equal')

        return fig, ax, cb
    
    @staticmethod
    def xy_to_pil_pixel_indices(image: Image.Image, lower_bounds: np.ndarray, upper_bounds: np.ndarray, coordinates, crop_to_image=False):
        pixel_widths = (upper_bounds - lower_bounds) / np.array(image.size)
        
        pixel_indices = np.floor_divide(coordinates - lower_bounds, pixel_widths).astype(int)

        if crop_to_image:
            in_bounds = np.all(np.logical_and(pixel_indices >= 0, pixel_indices < image.size), axis=-1)
            pixel_indices = pixel_indices[in_bounds, :]

        # flip the y-axis as PIL goes down instead of up
        pixel_indices[:, 1] = image.size[1] - pixel_indices[:, 1]
        return pixel_indices

    @staticmethod
    def draw_coordinate_frame(draw_image: ImageDraw.Draw, pixel_indices, line_kwargs):
        draw_indices = np.append(pixel_indices, pixel_indices[0:1, :], axis=0)

        # draw individual lines to buffer image
        for ii_line in range(len(draw_indices)-1):
            draw_image.line(
                (
                    draw_indices[ii_line, 0],
                    draw_indices[ii_line, 1],
                    draw_indices[ii_line+1, 0],
                    draw_indices[ii_line+1, 1],
                ),
                **line_kwargs
        )

    @staticmethod
    def draw_septum(
        t_values, rmean_per_t, h_per_t, s_grid,
        image: Image.Image, extent:tuple[float, float, float, float],
        line_kwargs,
        return_snapshots:bool=False, snapshot_timesteps:np.ndarray=None
    ):
        if return_snapshots and snapshot_timesteps is None:
            raise ValueError("Specify snapshot timesteps when returning snapshots")
        
        coords_per_s_per_t = ClosureODE.xy_from_rh(rmean_per_t, h_per_t, s_grid)

        x_min, x_max, y_min, y_max = extent
        n_x, n_y = image.size

        buffer_image = Image.new('L', (n_x, n_y), color='black')
        buffer_draw = ImageDraw.Draw(buffer_image)

        if return_snapshots:
            last_snapshot_index = 0
            snapshots = []

        for ii_t in range(len(t_values)):
            # clear the buffer
            buffer_draw.rectangle([0, 0, n_x, n_y], fill=0)
            
            # load pixel coordinates and add last element to close the loop
            pixel_coords = Plots.xy_to_pil_pixel_indices(image, np.array([x_min, y_min]), np.array([x_max, y_max]), coords_per_s_per_t[:, :, ii_t])
            # draw on buffer image
            Plots.draw_coordinate_frame(buffer_draw, pixel_coords, line_kwargs)

            # add buffer image to main image
            image = ImageChops.add(image, buffer_image)

            if return_snapshots and (last_snapshot_index < len(snapshot_timesteps)) and (t_values[ii_t] >= snapshot_timesteps[last_snapshot_index]):
                snapshots.append(np.array(image.copy()))
                last_snapshot_index += 1

        if not return_snapshots:
            return image
        return image, snapshots
    
class SeptationAnimator:
    def __init__(
            self, 
            fig, ax, 
            t_values, rmean_per_t, h_per_t, s_grid,
            image: Image.Image=None, extent:tuple[float, float, float, float]=None, line_kwargs:dict=None,
            snapshot_timesteps=None
        ):
        self.fig = fig
        self.ax = ax

        if extent is None:
            extent = (-1600, 1600, -1600, 1600)
        self.extent = extent
        
        if image is None:
            image = Image.new('L', (100, 100), color='black')
        self.image = image

        if snapshot_timesteps is None:
            snapshot_timesteps = np.arange(0, np.max(t_values))
        self.snapshot_timesteps = snapshot_timesteps

        _, self.snapshots = Plots.draw_septum(
            t_values, rmean_per_t, h_per_t, s_grid,
            image=image, extent=extent, line_kwargs=line_kwargs,
            return_snapshots=True, snapshot_timesteps=self.snapshot_timesteps
        )

        self.snapshots = np.array(self.snapshots)
        self.initialise_figure()
    
    def initialise_figure(self):
        self.txt = self.ax.text(0.05, 0.95, f't={self.snapshot_timesteps[0]}', transform=self.ax.transAxes, color='white', bbox=dict(facecolor='black'))
        self.im = self.ax.imshow(self.snapshots[0], interpolation=None, cmap='gray', extent=self.extent, vmin=np.min(self.snapshots), vmax=np.max(self.snapshots))

    def noisen_snapshots(self, noise_scheme='stepwise', noise_mean=128, noise_std=3):
        if noise_scheme == 'stepwise':
            rng = np.random.default_rng()
            for ii_snap in range(len(self.snapshots)):
                self.snapshots[ii_snap] += rng.normal(noise_mean, noise_std, size=self.snapshots[ii_snap].shape).astype(np.uint8)
        elif noise_scheme == 'constant':
            rng = np.random.default_rng()
            noise = rng.normal(noise_mean, noise_std, size=self.snapshots[0].shape).astype(np.uint8)
            for ii_snap in range(len(self.snapshots)):
                self.snapshots[ii_snap] += rng.normal(noise_mean, noise_std, size=self.snapshots[ii_snap].shape).astype(np.uint8)

        self.initialise_figure()

    def update(self, frame):
        # self.im.set_data(np.zeros_like(self.snapshots[frame]))
        self.im.set_data(self.snapshots[frame])
        self.txt.set_text(f't={self.snapshot_timesteps[frame]}s')

    def get_animation(self, interval: int=200):
        return FuncAnimation(fig=self.fig, func=self.update, frames=len(self.snapshot_timesteps), interval=interval)

def euler_step(func, t, y, stepsize):
    return y + (stepsize * func(t, y))

def constrained_euler_step(func, t, y, stepsize):
    return np.maximum(0., euler_step(func, t, y, stepsize))

def solve_euler(func, y_zero, stepsize, tmin, tmax, stepfunc: callable=None):
    if stepfunc is None:
        stepfunc = euler_step
    
    y_zero = np.array(y_zero)
    t_values = np.arange(tmin, tmax, stepsize)
    y_values = np.zeros(y_zero.shape + (len(t_values), ), dtype=y_zero.dtype)
    y_values[..., 0] = y_zero

    for ii_t, t_val in enumerate(t_values[1:]):
        y_values[..., ii_t+1] = stepfunc(func, t_val, y_values[..., ii_t], stepsize)

    return t_values, y_values