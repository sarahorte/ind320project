if 'wind_direction_10m (°)' in df_sel.columns:
            arrow_length = 0.05  # fraction of axis (vertical)
            total_days = (df_sel['time'].dt.date.max() - df_sel['time'].dt.date.min()).days + 1
            max_arrows = 31  # max arrows to avoid clutter

            if start_month == end_month:
                # Single month: one arrow per day (daily mean)
                df_daily = df_sel.groupby(df_sel['time'].dt.date)['wind_direction_10m (°)'].mean().reset_index() # daily mean
                for _, row in df_daily.iterrows(): # row has 'time' (date) and 'wind_direction_10m (°)'
                    deg = row['wind_direction_10m (°)'] # mean direction in degrees
                    rad = np.deg2rad(deg) # convert to radians
                    ax.annotate( # add wind direction arrows
                        '', # no text
                        xy=(row['time'], 0.5 + arrow_length * np.cos(rad)), # arrow tip
                        xytext=(row['time'], 0.5), # arrow base 
                        xycoords=('data', 'axes fraction'), # coordinates
                        textcoords=('data', 'axes fraction'), # coordinates 
                        arrowprops=dict(facecolor='k', edgecolor='k', width=1, headwidth=4, headlength=6) # arrow style
                    )
            else:
                # Multiple months: evenly spaced arrows, average wind direction for each segment
                num_arrows = min(total_days, max_arrows)
                dates = pd.date_range(df_sel['time'].dt.date.min(), df_sel['time'].dt.date.max(), periods=num_arrows)
                for date in dates:
                    # Take all rows on that day and compute mean
                    mask = df_sel['time'].dt.date == date.date()
                    if mask.any():
                        mean_deg = df_sel.loc[mask, 'wind_direction_10m (°)'].mean()
                    else:
                        mean_deg = df_sel['wind_direction_10m (°)'].mean()
                    rad = np.deg2rad(mean_deg)
                    ax.annotate(
                        '',
                        xy=(date, 0.5 + arrow_length * np.cos(rad)),
                        xytext=(date, 0.5),
                        xycoords=('data', 'axes fraction'),
                        textcoords=('data', 'axes fraction'),
                        arrowprops=dict(facecolor='k', edgecolor='k', width=1, headwidth=4, headlength=6)





    # --- Wind direction arrows only ---
    if choice == "All" or choice == "wind_direction_10m (°)":
        # Wind direction arrows averaged to one per day, with a maximum number of arrows. If there is a subset of more than one month, downsample to max_arrows. average direction per day included in the arrow.
        max_arrows = 31  # maximum number of arrows to display
        df_sel['date'] = df_sel['time'].dt.date
        df_daily = df_sel.groupby('date').agg({'wind_direction_10m (°)': 'mean', 'time': 'first'}).reset_index()
        if len(df_daily) > max_arrows:
            df_daily = df_daily.iloc[::len(df_daily)//max_arrows + 1]  # downsample to max_arrows 
        y_center = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2 if ax.get_ylim()[0] < ax.get_ylim()[1] else 0  # Place arrows at the vertical center of the left y-axis or at 0 if y-limits are equal
        arrow_length = 0.5 * (ax.get_ylim()[1] - ax.get_ylim()[0]) / 10 if ax.get_ylim()[1] > ax.get_ylim()[0] else 0.5  # Arrow length scaled to y-axis range or default to 0.5
        for _, row in df_daily.iterrows():
            direction_deg = row['wind_direction_10m (°)']
            direction_rad = np.deg2rad(direction_deg)  # Convert degrees to radians for trigonometric functions
            dx = arrow_length * np.sin(direction_rad)  # Calculate x and y components
            dy = arrow_length * np.cos(direction_rad)
            ax.arrow(row['time'], y_center, dx, dy, head_width=0.3, head_length=0.3, fc='k', ec='k')  # Draw arrow
            # make the y-axis be the same y-axis a when selecting "All". from -15 to 25.
            ax.set_ylim(-15, 25)