import os
import hashlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def filter_by_season_percentage(df, season, threshold=0.05, per_flyway=False):
    """
    Filter species (optionally per flyway) if their observations in the target season 
    are below a threshold fraction of total observations.

    Parameters
    ----------
    df : pd.DataFrame
        Must have ['scientific_name', 'year', 'month'], and optionally 'flyway'.
    season_periods : list of int
        Months defining the target season.
    threshold : float
        Minimum fraction of observations in season to keep species-year(-flyway).
    per_flyway : bool, default False
        If True, filter is applied separately per flyway.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.
    """

    # Define season periods 
    if(season=="summer"):
        season_periods = [11, 12, 1]
    elif(season=="winter"):
        season_periods = [6, 7, 8]
    else:
        raise ValueError("Season must be \"summer\" or winter.")
        
    
    # Define grouping columns
    group_cols = ["scientific_name", "year"]
    if per_flyway:
        group_cols.append("flyway")
    
    # Total counts per group
    total_counts = df.groupby(group_cols).size().rename("total")
    
    # Counts within the target season
    season_counts = (
        df[df["month"].isin(season_periods)]
        .groupby(group_cols)
        .size()
        .rename("season_count")
    )
    
    # Merge counts
    counts_df = pd.concat([total_counts, season_counts], axis=1).fillna(0)
    
    # Compute season ratio
    counts_df["season_ratio"] = counts_df["season_count"] / counts_df["total"]
    
    # Identify groups below threshold
    bad_groups = counts_df[counts_df["season_ratio"] < threshold].index
    
    # Build mask to remove only the rows corresponding to the "bad" groups **and** in season
    mask = df.set_index(group_cols).index.isin(bad_groups) & df["month"].isin(season_periods)
    
    return df.loc[~mask].reset_index(drop=True)


def calculate_observation_effort_by_checklists(df_sed, group_cols):
    """
    Calculate observation effort based on the number of checklists in df_sed.

    Parameters
    ----------
    df_sed : pd.DataFrame
        Sampling Event Data (one row per checklist).
    group_cols : list of str
        Columns to group by when counting checklists (e.g., ["year", "week"] or ["week", "flyway"]).

    Returns
    -------
    pd.DataFrame
        DataFrame with group columns and an `observation_effort` column scaled 0.1–1.0.
    """
    # Count rows (checklists) per group
    effort_df = (
        df_sed.groupby(group_cols)
        .size()
        .reset_index(name="num_checklists")
    )

    # Min-max scale to 0.1–1.0
    min_val = effort_df["num_checklists"].min()
    max_val = effort_df["num_checklists"].max()

    if min_val == max_val:
        effort_df["observation_effort"] = 0.55
    else:
        effort_df["observation_effort"] = 0.1 + 0.9 * (
            (effort_df["num_checklists"] - min_val) / (max_val - min_val)
        )

    return effort_df.drop(columns="num_checklists")

def fill_isolated_zeros(group):
    # Work on a copy to avoid SettingWithCopyWarning
    freqs = group['freq_adjusted'].to_numpy(copy=True)
    
    for i in range(1, len(freqs)-1):
        prev_val, curr_val, next_val = freqs[i-1], freqs[i], freqs[i+1]
        
        # Treat very small numbers as zero
        if abs(curr_val) < 1e-8 and abs(prev_val) >= 1e-8 and abs(next_val) >= 1e-8:
            freqs[i] = (prev_val + next_val) / 2
    
    # Assign back to a copy of group
    group = group.copy()
    group['freq_adjusted'] = freqs
    return group


def zero_fill_and_frequency(df, sed_df, granularity="month", per_flyway=False):
    """
    Zero-fill missing observations for species from SED,
    then compute frequency trajectories per year and chosen granularity.

    Parameters
    ----------
    df : pd.DataFrame
        EBD subset with at least ['scientific_name', 'sampling_event_identifier', 'observation_date', 'flyway'] if per_flyway=True.
    sed_df : pd.DataFrame
        SED subset with at least ['sampling_event_identifier', 'observation_date', 'flyway'] if per_flyway=True.
    granularity : str
        'month' or 'week'. Determines the time period used for grouping.
    per_flyway : bool
        If True, computes frequency per species per flyway.
        If False, computes frequency per species globally.
    """
    # --- Prepare SED with chosen granularity ---
    sed_df = sed_df[['sampling_event_identifier', 'observation_date'] + (['flyway'] if per_flyway else [])].copy()
    sed_df['observation_date'] = pd.to_datetime(sed_df['observation_date'], errors='coerce')
    sed_df['year'] = sed_df['observation_date'].dt.year

    if granularity == "month":
        sed_df['period'] = sed_df['observation_date'].dt.month
    elif granularity == "week":
        sed_df['period'] = sed_df['observation_date'].dt.isocalendar().week.astype(int)
    else:
        raise ValueError("granularity must be 'month' or 'week'")

    # --- Prepare observed species dataframe ---
    cols_obs = ['scientific_name', 'sampling_event_identifier', 'observation_date']
    if per_flyway:
        cols_obs.append('flyway')

    df_obs = df[cols_obs].copy()
    df_obs['species_observed'] = True
    df_obs['year'] = pd.to_datetime(df_obs['observation_date'], errors='coerce').dt.year
    if granularity == "month":
        df_obs['period'] = pd.to_datetime(df_obs['observation_date']).dt.month
    else:
        df_obs['period'] = pd.to_datetime(df_obs['observation_date']).dt.isocalendar().week.astype(int)

    # --- Build all species x checklist combinations ---
    species_list = df['scientific_name'].unique()

    if per_flyway:
        all_combos = pd.MultiIndex.from_product(
            [species_list, sed_df['sampling_event_identifier'].unique(), sed_df['flyway'].unique()],
            names=['scientific_name', 'sampling_event_identifier', 'flyway']
        ).to_frame(index=False)
    else:
        all_combos = pd.MultiIndex.from_product(
            [species_list, sed_df['sampling_event_identifier'].unique()],
            names=['scientific_name', 'sampling_event_identifier']
        ).to_frame(index=False)

    # Merge with SED to get checklist dates/periods
    merge_keys = ['sampling_event_identifier'] + (['flyway'] if per_flyway else [])
    all_combos = all_combos.merge(
        sed_df[['sampling_event_identifier', 'observation_date', 'year', 'period'] + (['flyway'] if per_flyway else [])],
        on=merge_keys,
        how='left'
    )

    # Merge with observed species
    merge_keys = ['scientific_name', 'sampling_event_identifier'] + (['flyway'] if per_flyway else [])
    all_combos = all_combos.merge(
        df_obs[merge_keys + ['species_observed']],
        on=merge_keys,
        how='left'
    )

    # Fill NaN species_observed with False
    all_combos['species_observed'] = all_combos['species_observed'].astype('boolean').fillna(False)

    # --- Compute frequency trajectories ---
    group_keys = ['scientific_name', 'year', 'period'] + (['flyway'] if per_flyway else [])
    freq = (
        all_combos
        .groupby(group_keys)
        .agg(frequency=('species_observed', 'mean'))
        .reset_index()
        .sort_values(group_keys)
    )

    return all_combos, freq


def plot_frequency_trajectories_carousel(freq_df, species_info=None, save_dir="../pictures/carousel", ylim_padding=1.25, frequency_col = "frequency"):
    """
    Plot frequency trajectories for each species and generate an HTML carousel.
    """
    # Unique hash for this dataset
    hash_id = hashlib.md5(pd.util.hash_pandas_object(freq_df, index=True).values).hexdigest()[:8]
    carousel_id = f"carousel_{hash_id}"
    full_dir = os.path.join(save_dir, f"dataset_{hash_id}")
    os.makedirs(full_dir, exist_ok=True)

    # Determine granularity
    granularity = "week" if freq_df['period'].max() > 12 else "month"

    # Build name mapping if provided
    name_map = {}
    if species_info is not None:
        name_map = species_info.set_index('scientific_name')['common_name'].to_dict()

    species_list = freq_df['scientific_name'].dropna().unique()
    saved_files = []

    for species in species_list:
        df_sp = freq_df[freq_df['scientific_name'] == species].copy()
        df_sp = df_sp.sort_values(by=['year', 'period'])

        # Title
        common_name = name_map.get(
            species,
            df_sp['common_name'].dropna().iloc[0] if 'common_name' in df_sp and df_sp['common_name'].notna().any() else ""
        )
        title = f"{common_name} ({species})" if common_name else species

        # Y limit
        max_freq = df_sp[frequency_col].max()
        y_max = min(1, max_freq * ylim_padding)

        # Plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=df_sp,
            x='period', y=frequency_col, hue='year', marker='o', palette='tab20'
        )
        plt.title(f"Observation Frequency Trajectory\n{title}")
        plt.ylabel("Frequency of Observation")

        if granularity == "month":
            plt.xlabel("Month")
            plt.xticks(range(1, 13),
                       ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        else:
            plt.xlabel("ISO Week")
            plt.xticks(range(1, 54, 2))

        plt.ylim(0, y_max)
        plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        filename = f"{full_dir}/{species.replace(' ', '_')}_trajectory_{granularity}.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        saved_files.append(os.path.basename(filename))

    # HTML carousel with unique ID
    html_path = os.path.join(full_dir, "carousel.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Species Frequency Carousel</title>
<style>
.carousel-container {{
  position: relative;
  max-width: 90%;
  margin: auto;
}}
.carousel-slide {{
  display: none;
}}
.carousel-slide img {{
  width: 100%;
  border-radius: 8px;
}}
.prev, .next {{
  cursor: pointer;
  position: absolute;
  top: 50%;
  padding: 16px;
  color: white;
  font-weight: bold;
  font-size: 18px;
  transition: 0.6s ease;
  border-radius: 0 3px 3px 0;
  background-color: rgba(0,0,0,0.5);
}}
.next {{
  right: 0;
  border-radius: 3px 0 0 3px;
}}
.prev:hover, .next:hover {{
  background-color: rgba(0,0,0,0.8);
}}
</style>
</head>
<body>

<div id="{carousel_id}" class="carousel-container">
""")
        # Add slides
        for i, img_file in enumerate(saved_files, start=1):
            display_style = "block" if i == 1 else "none"
            f.write(f'<div class="carousel-slide" style="display:{display_style};">\n')
            f.write(f'<img src="{full_dir}/{img_file}" alt="Slide {i}">\n')
            f.write("</div>\n")

        # Navigation buttons
        f.write(f"""
<a class="prev">&#10094;</a>
<a class="next">&#10095;</a>
</div>

<script>
(function() {{
  let slideIndex = 1;
  const container = document.getElementById("{carousel_id}");
  const slides = container.getElementsByClassName("carousel-slide");
  const prevBtn = container.querySelector(".prev");
  const nextBtn = container.querySelector(".next");

  function showSlides(n) {{
    if (n > slides.length) {{ slideIndex = 1 }}
    if (n < 1) {{ slideIndex = slides.length }}
    for (let i = 0; i < slides.length; i++) {{
      slides[i].style.display = "none";
    }}
    slides[slideIndex - 1].style.display = "block";
  }}

  function plusSlides(n) {{
    showSlides(slideIndex += n);
  }}

  prevBtn.addEventListener("click", () => plusSlides(-1));
  nextBtn.addEventListener("click", () => plusSlides(1));

  showSlides(slideIndex);
}})();
</script>

</body>
</html>
""")

    print(f"Plots saved to {full_dir} and carousel generated at {html_path}")
    return hash_id

def process_and_plot_species(
    df_species,
    df_sed,
    species_info=None,
    season=None,
    granularity="week",
    save_dir="../pictures/carousel",
):
    df_species_filtered = filter_by_season_percentage(df_species, season)
    all_combos_weekly, freq_df_weekly = zero_fill_and_frequency(df_species_filtered, df_sed, granularity="week")
    
    df_effort_weekly = calculate_observation_effort_by_checklists(df_sed, ["year", "week"])
    df_effort_weekly  = df_effort_weekly.rename(columns={"week": "period"})

    # Merge on 'period' and 'year'
    freq_df_weekly_adjusted = freq_df_weekly.merge(
        df_effort_weekly[['year', 'period', 'observation_effort']],
        on=['year', 'period'],
        how='left'
    )
    
    # Calculate adjusted frequency
    freq_df_weekly_adjusted["freq_adjusted"] = (
        freq_df_weekly_adjusted["frequency"] * freq_df_weekly_adjusted["observation_effort"]
    )

    
    # Apply per species and year
    freq_df_weekly_adjusted = (
        freq_df_weekly_adjusted
        .sort_values(['scientific_name', 'year', 'period'])
        .groupby(['scientific_name', 'year'], group_keys=False)
        .apply(fill_isolated_zeros)
    )

    plot_frequency_trajectories_carousel(freq_df_weekly_adjusted, species_info=species_info, frequency_col="freq_adjusted")
    
    return freq_df_weekly_adjusted