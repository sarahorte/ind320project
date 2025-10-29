import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.neighbors import LocalOutlierFactor



# --- Cell: SPC outlier detection (DCT SATV) + Plotly plotting ---
import numpy as np
import pandas as pd
from scipy.fftpack import dct, idct
import plotly.graph_objects as go

# DCT-based seasonal decomposition and SATV calculation
def dct_seasonal_and_satv(series: pd.Series, cutoff_frac: float = 0.05):
    """Return seasonal component (low-frequency DCT reconstruction) and SATV = series - seasonal."""
    x = series.values.astype(float)
    n = len(x)
    X = dct(x, norm='ortho')
    keep = int(np.floor(cutoff_frac * n))
    if keep < 1:
        keep = 1
    seasonal_coeffs = np.zeros_like(X)
    seasonal_coeffs[:keep] = X[:keep]
    seasonal = idct(seasonal_coeffs, norm='ortho')
    satv = x - seasonal
    return pd.Series(seasonal, index=series.index), pd.Series(satv, index=series.index)

# Median Absolute Deviation
def mad(arr):
    med = np.median(arr)
    return np.median(np.abs(arr - med))

# SPC outlier detection and Plotly plotting function
def spc_outlier_plotly(temp_series: pd.Series, cutoff_frac: float = 0.05, k: float = 3.0, title: str = None):
    """
    temp_series: pandas Series with datetime index and temperature values (°C).
    cutoff_frac: fraction of lowest DCT frequencies to KEEP as seasonal (default 0.05; 0.04–0.08 are sensible range).
    k: number of robust standard deviations to use for SPC boundaries (default 3).
    Returns: (plotly_fig, summary_dict)
    """
    if not isinstance(temp_series.index, pd.DatetimeIndex):
        temp_series = temp_series.copy()
        temp_series.index = pd.to_datetime(temp_series.index)

    temp_series = temp_series.sort_index()
    seasonal, satv = dct_seasonal_and_satv(temp_series, cutoff_frac=cutoff_frac)

    # robust stats on SATV
    med = float(np.median(satv.values))
    mad_val = float(mad(satv.values))
    sigma = float(1.4826 * mad_val) if mad_val > 0 else float(np.std(satv.values))

    # SATV thresholds (constant values) and convert to original scale by adding seasonal component
    lower_satv = med - k * sigma
    upper_satv = med + k * sigma
    lower_curve = seasonal + lower_satv
    upper_curve = seasonal + upper_satv

    outlier_mask = (satv < lower_satv) | (satv > upper_satv)
    n_points = len(temp_series)
    n_outliers = int(outlier_mask.sum())
    outlier_fraction = n_outliers / n_points if n_points > 0 else 0.0

    # Build Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=temp_series.index, y=temp_series.values, mode='lines', name='Temperature'))
    fig.add_trace(go.Scatter(x=temp_series.index, y=upper_curve.values, mode='lines', name=f'Upper SPC (k={k})', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=temp_series.index, y=lower_curve.values, mode='lines', name=f'Lower SPC (k={k})', line=dict(dash='dash')))
    if n_outliers > 0:
        fig.add_trace(go.Scatter(x=temp_series.index[outlier_mask], y=temp_series.values[outlier_mask],
                                 mode='markers', name='Outliers', marker=dict(size=6, symbol='x')))

    fig.update_layout(
        title=title or 'Temperature and SPC outliers (DCT-based SATV)',
        xaxis_title='Time',
        yaxis_title='Temperature (°C)',
        hovermode='x unified',
        template='plotly_white',
        height=520,
        width=1100
    )

    summary = {
        'n_points': n_points, # total number of data points
        'n_outliers': n_outliers, # number of detected outliers
        'outlier_fraction': outlier_fraction, # fraction of outliers
        'median_satv': med, # median of SATV
        'mad_satv': mad_val, # MAD of SATV
        'sigma_est': sigma, # robust std dev estimate of SATV
        'cutoff_frac': float(cutoff_frac), # DCT cutoff fraction
        'k': float(k), # SPC k parameter
        'example_outlier_times': list(map(str, temp_series.index[outlier_mask][:20])) # first 20 outlier timestamps as strings
    }

    return fig, summary








def lof_precipitation_plotly(precip_series: pd.Series, contamination: float = 0.01, n_neighbors: int = 20, title: str = None):
    """
    Detect precipitation anomalies using the Local Outlier Factor (LOF) method and visualize with Plotly.

    Parameters
    ----------
    precip_series : pandas.Series
        Time series of precipitation (mm/hour or mm/day) with datetime index.
    contamination : float, optional
        Proportion of outliers (default 0.01 = 1%).
    n_neighbors : int, optional
        Number of neighbors to use for LOF (default 20).
    title : str, optional
        Custom plot title.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure of precipitation with anomalies marked.
    summary : dict
        Summary with counts, proportions, and example outlier timestamps.
    """

    # Ensure datetime index
    if not isinstance(precip_series.index, pd.DatetimeIndex):
        precip_series = precip_series.copy()
        precip_series.index = pd.to_datetime(precip_series.index)
    precip_series = precip_series.sort_index()

    # Prepare data for LOF (2D input required)
    X = precip_series.values.reshape(-1, 1)

    # Fit LOF model
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    preds = lof.fit_predict(X)
    lof_scores = -lof.negative_outlier_factor_

    # Identify outliers
    outlier_mask = preds == -1
    n_points = len(precip_series)
    n_outliers = int(outlier_mask.sum())
    outlier_fraction = n_outliers / n_points if n_points > 0 else 0.0

    # Build Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=precip_series.index, y=precip_series.values,
        mode='lines', name='Precipitation'
    ))

    if n_outliers > 0:
        fig.add_trace(go.Scatter(
            x=precip_series.index[outlier_mask],
            y=precip_series.values[outlier_mask],
            mode='markers',
            name='Anomalies (LOF)',
            marker=dict(color='red', size=6, symbol='x')
        ))

    fig.update_layout(
        title=title or f'Precipitation anomalies (LOF, contamination={contamination*100:.1f}%)',
        xaxis_title='Time',
        yaxis_title='Precipitation',
        hovermode='x unified',
        template='plotly_white',
        height=520,
        width=1100
    )

    summary = {
        'n_points': n_points,
        'n_outliers': n_outliers,
        'outlier_fraction': outlier_fraction,
        'contamination_param': contamination,
        'n_neighbors': n_neighbors,
        'lof_score_min': float(lof_scores.min()),
        'lof_score_max': float(lof_scores.max()) # max LOF score. LOF scores indicate the degree of outlierness; higher scores mean more anomalous.
        }

    return fig, summary



import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import STL

def stl_decomposition_plotly_subplots(
    df,
    price_area='NO1',
    production_group='hydro',
    period=24,
    seasonal=13,
    trend=25,
    robust=True
):
    """
    STL decomposition with four stacked subplots (Original, Trend, Seasonal, Residual) using Plotly.

    Returns the figure and STL results object.
    """
    # Case-insensitive filtering
    ts = df[(df['priceArea'].str.lower() == price_area.lower()) &
            (df['productionGroup'].str.lower() == production_group.lower())]['quantityKwh']
    
    if ts.empty:
        raise ValueError(f"No data found for price area '{price_area}' and production group '{production_group}'")
    
    # Fill missing values
    ts = ts.asfreq('h').ffill()
    
    # Fit STL
    stl = STL(ts, period=period, seasonal=seasonal, trend=trend, robust=robust)
    res = stl.fit()
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual']
    )
    
    # Original
    fig.add_trace(go.Scatter(x=ts.index, y=ts, mode='lines', name='Original', line=dict(color='blue')), row=1, col=1)
    # Trend
    fig.add_trace(go.Scatter(x=ts.index, y=res.trend, mode='lines', name='Trend', line=dict(color='orange')), row=2, col=1)
    # Seasonal
    fig.add_trace(go.Scatter(x=ts.index, y=res.seasonal, mode='lines', name='Seasonal', line=dict(color='green')), row=3, col=1)
    # Residual
    fig.add_trace(go.Scatter(x=ts.index, y=res.resid, mode='lines', name='Residual', line=dict(color='red')), row=4, col=1)
    
    fig.update_layout(
        height=900,
        title_text=f"STL Decomposition: {production_group} in {price_area}",
        template='plotly_white'
    )
    
    fig.show()
    
    return fig, res





import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

def matplotlib_spectrogram(
    df,
    price_area='NO1',
    production_group='hydro',
    window_length=168,   # nperseg in STFT
    window_overlap=84,   # noverlap in STFT
    fs=1                 # sampling frequency (1 per hour)
):
    """
    Compute and plot a spectrogram for electricity production data using Matplotlib.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ['priceArea','productionGroup','quantityKwh'] indexed by datetime.
    price_area : str
        Electricity price area to filter.
    production_group : str
        Production group to filter.
    window_length : int
        Number of samples per STFT window (nperseg).
    window_overlap : int
        Overlap between windows (noverlap).
    fs : float
        Sampling frequency. For hourly data, fs=1.

    Returns
    -------
    f : np.ndarray
        Frequency bins.
    t : np.ndarray
        Time bins.
    Zxx : np.ndarray
        STFT complex values.
    """
    # Filter data
    ts = df[(df['priceArea'].str.lower() == price_area.lower()) &
            (df['productionGroup'].str.lower() == production_group.lower())]['quantityKwh']
    
    if ts.empty:
        raise ValueError(f"No data for price area '{price_area}' and production group '{production_group}'")
    
    ts = ts.asfreq('h').ffill()
    
    # Compute STFT
    f, t, Zxx = stft(ts.values, fs=fs, nperseg=window_length, noverlap=window_overlap)
    
    # Plot with Matplotlib
    plt.figure(figsize=(12, 5))
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='viridis', vmin=0, vmax=np.max(np.abs(Zxx)))
    plt.title(f'Spectrogram: {production_group} in {price_area}')
    plt.ylabel('Frequency [1/hour]')
    plt.xlabel('Time [hours]')
    plt.colorbar(label='Amplitude')
    plt.tight_layout()
    plt.show()
    
    return f, t, Zxx

