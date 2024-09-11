import contextlib
import pickle
import matplotlib
import warnings
from matplotlib import (
    pyplot as plt, 
    figure as mpl_fig, 
    axes as mpl_ax,
    image as mpl_img,
    gridspec as mpl_gridspec)
from numpy.typing import ArrayLike
from typing import Generator, Tuple, Any, Optional, Literal, List


_DEFAULT_FIGURE_SETTINGS = {
    'mathtext.fontset': 'cm',  # Allow math font for LaTeX expressions
    'font.size': 11,
    'lines.linewidth': 2,
    'axes.titlepad': 5,
}

_HALF_WIDTH_PAPER_FIGURE_SETTINGS = {
    'figure.figsize': (5.5, 4.125),
    **_DEFAULT_FIGURE_SETTINGS
}

_FULL_WIDTH_PAPER_FIGURE_SETTINGS = {
    'figure.figsize': (11, 8.25),
    **_DEFAULT_FIGURE_SETTINGS,
}


def _to_string(v: Any) -> str:
    if isinstance(v, (int, float)):
        return str(v).replace('-', '\N{MINUS SIGN}')
    return str(v)


def to_ticklabel(values: List[Any]) -> List[str]:
    return [_to_string(v) for v in values]


def dump_figures(fig, ax, rc_settings, path: str):
    with open(path, 'wb') as f_pickle:
        pickle.dump({
            'fig': fig,
            'ax': ax,
            'rc_settings': rc_settings,
            'matplotlib.__version__': matplotlib.__version__
        }, f_pickle)


@contextlib.contextmanager
def load_figures(
        path: str, 
        save_path: Optional[str] = None, 
        show: Optional[bool] = True
) -> Generator[Tuple[mpl_fig.Figure, Any], None, None]:

    with open(path, 'rb') as f_pickle:
        pickle_obj = pickle.load(f_pickle)
        if pickle_obj['matplotlib.__version__'] != matplotlib.__version__:
            warnings.warn('Mismatch matplotlib version! Serialisation may fail!')

        loaded_fig = pickle_obj['fig']
        loaded_ax = pickle_obj['ax']
        loaded_rc_settings = pickle_obj['rc_settings']

        with plt.rc_context(loaded_rc_settings):
            # "Steal" canvas from a dummy figure
            dummy_fig = plt.figure()
            new_manager = dummy_fig.canvas.manager
            # Connect manager and figure
            new_manager.canvas.figure = loaded_fig
            loaded_fig.set_canvas(new_manager.canvas)
        
            try:
                yield (loaded_fig, loaded_ax)
                loaded_fig.tight_layout()
            finally:
                if show:
                    plt.show()
                if save_path is not None:
                    if save_path.endswith('.pickle'):
                        dump_figures(loaded_fig, loaded_ax, loaded_rc_settings, path=save_path)
                    else:
                        loaded_fig.savefig(fname=save_path)
                plt.close(fig=loaded_fig)
                plt.close(fig=dummy_fig)
                

@contextlib.contextmanager
def default_subplots(
        save_path: Optional[str] = None,
        show: Optional[bool] = True,
        paper_mode: Literal['halfwidth', 'fullwidth'] = 'halfwidth',
        **kwargs
) -> Generator[Tuple[mpl_fig.Figure, Any], None, None]:
    if paper_mode == 'halfwidth':
        rc_settings = _HALF_WIDTH_PAPER_FIGURE_SETTINGS
    elif paper_mode == 'fullwidth':
        rc_settings = _FULL_WIDTH_PAPER_FIGURE_SETTINGS
    with plt.rc_context(rc_settings):
        fig, ax = plt.subplots(**kwargs)

        try:
            yield (fig, ax)
            fig.tight_layout()
        finally:
            if show:
                plt.show()
            if save_path is not None:
                if save_path.endswith('.pickle'):
                    dump_figures(fig, ax, rc_settings, path=save_path)
                else:
                    fig.savefig(fname=save_path)
            plt.close(fig=fig)


def imshow_log_spec(
        ax: mpl_ax.Axes,
        X_ft: ArrayLike,
        interpolation: Optional[str] = 'none',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        **kwargs) -> mpl_img.AxesImage:
    return ax.imshow(
        X_ft,
        cmap='jet',
        aspect='auto',
        interpolation=interpolation,
        vmin=vmin,
        vmax=vmax,
        origin='lower',
        **kwargs)


def imshow_binary_mask(
        ax: mpl_ax.Axes,
        X_ft: ArrayLike,
        interpolation: Optional[str] = 'none',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        **kwargs) -> mpl_img.AxesImage:
    return ax.imshow(
        X_ft,
        cmap='binary',
        aspect='auto',
        interpolation=interpolation,
        vmin=vmin,
        vmax=vmax,
        origin='lower',
        **kwargs)


def imshow_angular_spec(
        ax: mpl_ax.Axes,
        X_ft: ArrayLike,
        interpolation: Optional[str] = 'none',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        **kwargs) -> mpl_img.AxesImage:
    return ax.imshow(
        X_ft,
        cmap='twilight_shifted',
        aspect='auto',
        interpolation=interpolation,
        vmin=vmin,
        vmax=vmax,
        origin='lower',
        **kwargs)


@contextlib.contextmanager
def default_gridspec_subplots(
        save_path: Optional[str] = None,
        show: Optional[bool] = True,
        paper_mode: Literal['halfwidth', 'fullwidth'] = 'halfwidth',
        nrows: int = 1,
        ncols: int = 1,
        width_ratios: Optional[ArrayLike] = None,
        height_ratios: Optional[ArrayLike] = None,
        **kwargs
) -> Generator[Tuple[mpl_fig.Figure, Any], None, None]:
    if paper_mode == 'halfwidth':
        rc_settings = _HALF_WIDTH_PAPER_FIGURE_SETTINGS
    elif paper_mode == 'fullwidth':
        rc_settings = _FULL_WIDTH_PAPER_FIGURE_SETTINGS

    with plt.rc_context(rc_settings):
        fig = plt.figure(**kwargs)
        gs = mpl_gridspec.GridSpec(nrows=nrows, ncols=ncols, width_ratios=width_ratios, height_ratios=height_ratios)

        try:
            yield (fig, gs)
            # fig.tight_layout()
            gs.tight_layout(figure=fig)
        finally:
            if show:
                plt.show()
            if save_path is not None:
                fig.savefig(fname=save_path)
            plt.close(fig=fig)