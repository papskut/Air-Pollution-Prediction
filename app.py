import streamlit as st
import pandas as pd
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Dashboard Polusi & Kesehatan Global",
    page_icon="💨",
    layout="wide"
)

@st.cache_resource
def load_model():
    try: return joblib.load('pollution_model.pkl')
    except Exception as e: st.error(f"Error memuat model: {e}"); return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('cleaned_data_for_app.csv')
        filter_data = {'countries': sorted(df['Country'].unique().tolist())}
        
        region_col = next((col for col in ['Name_pollution', 'Region'] if col in df.columns), None)
        if not region_col: raise KeyError("Kolom Region ('Name_pollution' atau 'Region') tidak ditemukan!")
        
        filter_data['region_column'] = region_col
        filter_data['regions'] = sorted(df[region_col].unique().tolist())

        filter_data['iso_mapping'] = {} 
        if 'ISO3' in df.columns and 'Country' in df.columns:
            mapping_subset = df[['Country', 'ISO3']].copy() 
            mapping_subset.drop_duplicates(inplace=True)
            if isinstance(mapping_subset, pd.DataFrame) and not mapping_subset.empty:
                try:
                    mapping_subset.set_index('Country', inplace=True)
                    if 'ISO3' in mapping_subset.columns:
                            filter_data['iso_mapping'] = mapping_subset['ISO3'].astype(str).to_dict()
                except KeyError:
                        st.warning("Gagal set index 'Country' saat membuat ISO mapping.")
                except Exception as e_map:
                        st.warning(f"Error saat membuat ISO mapping: {e_map}")
            else:
                st.warning("Data subset untuk ISO mapping kosong atau bukan DataFrame.")
        else:
            st.warning("Kolom 'ISO3' atau 'Country' tidak ditemukan. Peta mungkin tidak berfungsi.")

        if not filter_data['iso_mapping']: st.warning("ISO mapping kosong. Peta mungkin tidak berfungsi.")
        return df, filter_data
    
    except FileNotFoundError: st.error("File data 'cleaned_data_for_app.csv' tidak ditemukan."); return None, None
    except Exception as e: st.error(f"Error memuat data: {e}"); return None, None


@st.cache_data
def get_default_exposure(target_years: list, historical_data: pd.DataFrame, time_col: str = 'Year', value_col: str = 'Exposure Mean'):
    num_years = len(target_years) if isinstance(target_years, list) else 1
    if time_col not in historical_data.columns or value_col not in historical_data.columns:
        return [25.0] * num_years 
    
    clean_data = historical_data[[time_col, value_col]].dropna()
    
    if len(clean_data) < 5:
        return [25.0] * num_years 

    try:
        X_hist, y_hist = clean_data[[time_col]], clean_data[value_col]
        trend_model = LinearRegression().fit(X_hist, y_hist)
        
        predicted_exposures = trend_model.predict(pd.DataFrame({time_col: target_years}))
        
        if isinstance(predicted_exposures, (pd.Series, pd.DataFrame)): 
            predicted_exposures = predicted_exposures.values

        clipped_values = np.clip(predicted_exposures, 0, 120) 
        return clipped_values.tolist() 
    
    except Exception as e: 
        st.warning(f"Error calculating trend: {e}") 
        return [25.0] * num_years 


model = load_model()
df, filter_data = load_data()
if model is None or df is None or filter_data is None: st.stop()

st.title("Dashboard Dampak Polusi Udara & Kesehatan Global 💨")

tab_list = ["📈 Dashboard Utama", "🗺️ Peta Interaktif", "📊 Prediksi Model", "🗃️ Data Mentah"]
tab1, tab2, tab3, tab4 = st.tabs(tab_list)

with tab1:
    st.markdown("### Filter Data Dashboard")
    col_filter1, col_filter2 = st.columns(2)
    region_col_name = filter_data['region_column']
    
    with col_filter1:
        regions = ["Global"] + filter_data['regions']
        selected_region = st.selectbox("Pilih Region:", regions, key="tab1_global_region")
    with col_filter2:
        countries_list = ["Semua Negara"]
        if selected_region == "Global": countries_list += filter_data['countries']
        elif region_col_name in df.columns: countries_list += sorted(df[df[region_col_name] == selected_region]['Country'].unique().tolist())
        selected_country = st.selectbox("Pilih Negara:", countries_list, key="tab1_global_country")
    
    df_filtered, display_name = df, "Global"
    try:
        if selected_country != "Semua Negara":
            df_filtered = df[df['Country'] == selected_country]
            display_name = selected_country
        elif selected_region != "Global" and region_col_name in df.columns:
            df_filtered = df[df[region_col_name] == selected_region]
            display_name = f"Region: {selected_region}"
    except Exception as e: st.error(f"Error filter: {e}"); df_filtered = df 
    st.markdown("---")

    st.header(f"Dashboard untuk: {display_name}")
    if df_filtered.empty: st.warning("Tidak ada data untuk filter.")
    else:
        st.subheader("Statistik Rata-rata (1990-2020)")
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        avg_death = df_filtered['Burden Mean'].mean()
        avg_exposure = df_filtered['Exposure Mean'].mean()
        col_kpi1.metric("Rata-rata Kematian / Tahun", f"{avg_death:,.0f}")
        col_kpi2.metric("Rata-rata Polusi (µg/m³)", f"{avg_exposure:,.2f}")
        col_kpi3.metric("Total Data Poin", f"{len(df_filtered):,}")
        st.markdown("---")
        st.subheader("Tren Historis (1990-2020)")
        df_trend = df_filtered.groupby('Year', as_index=False)[['Burden Mean', 'Exposure Mean']].mean()
        if not df_trend.empty:
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=df_trend['Year'], y=df_trend['Burden Mean'], name='Kematian', line=dict(color='#636EFA')))
            fig_trend.add_trace(go.Scatter(x=df_trend['Year'], y=df_trend['Exposure Mean'], name='Polusi', line=dict(color='#00CC96'), yaxis='y2'))
            fig_trend.update_layout(title_text="Tren Polusi vs Kematian Tahunan", xaxis_title="Tahun",
                                    yaxis=dict(title="Rata-rata Kematian", title_font=dict(color="#636EFA"), tickfont=dict(color="#636EFA")),
                                    yaxis2=dict(title="Rata-rata Polusi", title_font=dict(color="#00CC96"), tickfont=dict(color="#00CC96"), overlaying='y', side='right'),
                                    legend=dict(x=0.1, y=1.1, orientation="h"))
            st.plotly_chart(fig_trend, use_container_width=True)
        else: st.warning("Data tren tidak cukup.")
        st.subheader(f"Analisis Korelasi untuk: {display_name}")
        if len(df_filtered) > 1:
                df_sample = df_filtered.sample(min(1000, len(df_filtered)))
                try:
                    fig_scatter = px.scatter(df_sample, x="Exposure Mean", y="Burden Mean", title=f"Korelasi Polusi vs Kematian di {display_name}", trendline="ols", hover_name="Country", hover_data=["Year"])
                    fig_scatter.update_layout(height=500)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                except Exception as e: st.warning(f"Gagal membuat plot korelasi: {e}")
        else: st.warning("Data tidak cukup untuk plot korelasi.")
        st.markdown("---")
        if display_name == "Global":
            st.subheader("Peringkat Global (Rata-rata 1990-2020)")
            col_chart1, col_chart2 = st.columns(2)
            if not df.empty: 
                with col_chart1:
                    top_10_death = df.groupby('Country', as_index=False)['Burden Mean'].mean().nlargest(10, 'Burden Mean')
                    if not top_10_death.empty:
                        fig_top_death = px.bar(top_10_death.sort_values('Burden Mean'), y='Country', x='Burden Mean', orientation='h', title="Top 10 Negara: Kematian Tertinggi")
                        st.plotly_chart(fig_top_death, use_container_width=True)
                with col_chart2:
                    top_10_exp = df.groupby('Country', as_index=False)['Exposure Mean'].mean().nlargest(10, 'Exposure Mean')
                    if not top_10_exp.empty:
                        fig_top_exp = px.bar(top_10_exp.sort_values('Exposure Mean'), y='Country', x='Exposure Mean', orientation='h', title="Top 10 Negara: Polusi Tertinggi")
                        st.plotly_chart(fig_top_exp, use_container_width=True)

with tab2:
    st.header("Peta Sebaran Kematian Akibat Polusi")
    col_map_slider, col_map_input = st.columns([3, 1])
    
    if 'map_year' not in st.session_state: st.session_state.map_year = 2020
    if 'map_exposure' not in st.session_state: st.session_state.map_exposure = None 
    
    def update_map_year():
        st.session_state.map_year = st.session_state.map_year_slider_key
        st.session_state.map_exposure = None 
            
    with col_map_slider:
        st.slider("Pilih Tahun:", min_value=1990, max_value=2045, 
                    value=st.session_state.map_year, step=1, 
                    key="map_year_slider_key", on_change=update_map_year) 
    
    current_map_year = st.session_state.map_year
    map_data_type = "Historis" if current_map_year <= 2020 else "Prediksi"
    future_exposure_map_value = None
    
    if map_data_type == "Prediksi":
            with col_map_input:
                df_global_hist = df.groupby('Year', as_index=False)['Exposure Mean'].mean()
                default_exp_global_list = get_default_exposure(target_years=[current_map_year], historical_data=df_global_hist)
                default_exp_global = default_exp_global_list[0] if default_exp_global_list else 25.0
                
                if st.session_state.map_exposure is None:
                    st.session_state.map_exposure = default_exp_global
                
                def update_map_exposure_manual(): 
                    st.session_state.map_exposure = st.session_state.map_future_exposure_key
                
                st.number_input(f"Asumsi Polusi Global ({current_map_year}):", min_value=0.0, max_value=120.0, 
                                value=round(st.session_state.map_exposure, 1), step=0.1, 
                                key="map_future_exposure_key", 
                                on_change=update_map_exposure_manual, 
                                help="Nilai default dihitung dari tren global.")
                
                future_exposure_map_value = st.session_state.map_exposure 

    def create_map_figure(year, is_prediction, exposure_assumption=None):
        df_map_data, map_title, color_col, hover_dict = pd.DataFrame(), f"Data Peta Tahun {year}", 'log_Burden', {"ISO3": False, "log_Burden": False}
        iso_mapping = filter_data.get('iso_mapping', {})
        if not iso_mapping: return None 
        if is_prediction:
            if exposure_assumption is None: return None 
            all_countries = filter_data.get('countries', [])
            if not all_countries: return None
            pred_map_input = pd.DataFrame({'Exposure Mean': [exposure_assumption] * len(all_countries),'Year': [year] * len(all_countries), 'Country': all_countries})
            try:
                predicted_deaths = model.predict(pred_map_input)
                df_map_data = pd.DataFrame({'Country': all_countries, 'Burden Mean': predicted_deaths, 'Year': year})
                df_map_data['Burden Mean'] = df_map_data['Burden Mean'].clip(lower=0) 
                df_map_data['ISO3'] = df_map_data['Country'].map(iso_mapping)
                df_map_data.dropna(subset=['ISO3'], inplace=True) 
                if df_map_data.empty: return None
                df_map_data['log_Burden'] = np.log10(df_map_data['Burden Mean'].clip(lower=1)) 
                hover_dict["Burden Mean"] = ":,.0f" 
                map_title = f"Angka Kematian (Prediksi) Tahun {year}"
            except Exception as e: st.error(f"Error prediksi peta: {e}"); return None
        else: 
            df_map_data = df[df['Year'] == year].copy()
            if df_map_data.empty: return None
            df_map_data['log_Burden'] = np.log10(df_map_data['Burden Mean'].clip(lower=1)) 
            hover_dict["Burden Mean"] = ":,.0f" 
            map_title = f"Angka Kematian (Historis) Tahun {year}"
        
        if not df_map_data.empty and 'ISO3' in df_map_data.columns and color_col in df_map_data.columns:
            fig = px.choropleth(df_map_data, locations="ISO3", color=color_col, hover_name="Country", hover_data={"Burden Mean": ":,.0f"}, color_continuous_scale="Plasma", title=map_title)
            fig.update_layout(height=600, legend_title_text='Jumlah Kematian (Skala Log)')
            return fig
        else: return None
    
    fig_map = create_map_figure(current_map_year, map_data_type == "Prediksi", future_exposure_map_value)
    
    if fig_map: st.plotly_chart(fig_map, use_container_width=True)
    elif map_data_type == "Prediksi" and future_exposure_map_value is None: 
        st.info("Memproses asumsi polusi...") 
    else: st.warning(f"Tidak ada data peta untuk tahun {current_map_year}.")

with tab3:
    st.header("Prediksi Model Masa Depan")
    st.write("Prediksi masa depan untuk **satu negara** menggunakan tren polusi otomatis.")
    col_form, col_result = st.columns([1, 2])
    
    if 'pred_country' not in st.session_state: st.session_state.pred_country = filter_data['countries'][0]
    if 'pred_year' not in st.session_state: st.session_state.pred_year = 2030
    if 'pred_exposure_auto_list' not in st.session_state: st.session_state.pred_exposure_auto_list = [] 
    if 'pred_show_results' not in st.session_state: st.session_state.pred_show_results = False
    
    with col_form:
        st.subheader("Input Prediksi")
        current_year = 2025
        def on_change_pred_country_final(): 
                st.session_state.pred_country = st.session_state.pred_country_widget_key_final
                st.session_state.pred_show_results = False 
        default_country_index = filter_data['countries'].index(st.session_state.pred_country)
        st.selectbox("Pilih Negara:", filter_data['countries'], index=default_country_index, key="pred_country_widget_key_final", on_change=on_change_pred_country_final) 
        country_hist_data = df[df['Country'] == st.session_state.pred_country] 
        
        with st.form("prediction_form_key_final"):
            selected_future_year = st.number_input(f"Prediksi Hingga Tahun:", min_value=current_year, max_value=current_year + 20, value=st.session_state.pred_year, key="pred_year_widget_key_final")
            
            use_auto_exposure = True 
            
            future_years_list_for_auto = []
            calculated_default_exposures_list = []
            if not country_hist_data.empty:
                last_hist_year = int(country_hist_data['Year'].max())
                future_years_list_for_auto = list(range(last_hist_year + 1, selected_future_year + 1))
                if future_years_list_for_auto: 
                    calculated_default_exposures_list = get_default_exposure(target_years=future_years_list_for_auto, historical_data=country_hist_data)

            calculated_default_exposure_final_year = calculated_default_exposures_list[-1] if calculated_default_exposures_list else 25.0
            
            st.caption(f"Tren polusi otomatis dihitung per tahun (Nilai akhir ~ {calculated_default_exposure_final_year:.1f})")
            
            submit_button = st.form_submit_button("Jalankan Prediksi")
            if submit_button:
                st.session_state.pred_country = st.session_state.pred_country_widget_key_final 
                st.session_state.pred_year = selected_future_year
                st.session_state.pred_exposure_list_final = calculated_default_exposures_list 
                st.session_state.pred_show_results = True 
    
    with col_result:
        st.subheader("Hasil Prediksi")
        if st.session_state.pred_show_results: 
            pred_country_to_use = st.session_state.pred_country
            future_year_to_use = st.session_state.pred_year
            exposure_values_to_use_list = st.session_state.pred_exposure_list_final
            
            try:
                history_data = df[df['Country'] == pred_country_to_use][['Year', 'Burden Mean', 'Exposure Mean']].sort_values('Year').copy()
                if history_data.empty: st.error(f"Data historis kosong.")
                else:
                    last_data_year = history_data['Year'].max()
                    last_burden = history_data.iloc[-1]['Burden Mean']
                    last_exposure = history_data.iloc[-1]['Exposure Mean']
                    future_years = list(range(int(last_data_year) + 1, future_year_to_use + 1))
                    
                    if not future_years: st.error(f"Tahun prediksi salah.")
                    elif len(exposure_values_to_use_list) != len(future_years): st.error(f"Error internal: Jumlah polusi ({len(exposure_values_to_use_list)}) != thn ({len(future_years)}).")
                    else:
                        pred_df = pd.DataFrame({'Exposure Mean': exposure_values_to_use_list, 'Year': future_years, 'Country': [pred_country_to_use] * len(future_years)})
                        future_predictions = model.predict(pred_df).clip(min=0) 
                        future_data = pd.DataFrame({'Year': future_years, 'Burden Mean': future_predictions, 'Exposure Mean': exposure_values_to_use_list})
                        
                        if future_data.empty: st.warning("Tidak ada data prediksi.")
                        else:
                            pred_value_final_burden = future_data['Burden Mean'].iloc[-1]
                            pred_value_final_exposure = future_data['Exposure Mean'].iloc[-1]
                            st.metric(label=f"Prediksi Kematian {pred_country_to_use} thn {future_year_to_use}", value=f"{pred_value_final_burden:,.0f}", delta=f"{pred_value_final_burden - last_burden:,.0f} vs {int(last_data_year)}")
                            
                            exp_label = "Prediksi Polusi (Otomatis)"
                            st.metric(label=f"{exp_label} {pred_country_to_use} thn {future_year_to_use}", value=f"{pred_value_final_exposure:,.2f}", delta=f"{pred_value_final_exposure - last_exposure:,.2f} vs {int(last_data_year)}")
                            
                            fig_pred_trend = go.Figure()
                            fig_pred_trend.add_trace(go.Scatter(x=history_data['Year'], y=history_data['Burden Mean'], name='Kematian (Hist)', line=dict(color='#636EFA', dash='solid'), mode='lines+markers'))
                            fig_pred_trend.add_trace(go.Scatter(x=history_data['Year'], y=history_data['Exposure Mean'], name='Polusi (Hist)', line=dict(color='#00CC96', dash='solid'), yaxis='y2', mode='lines+markers'))
                            if not future_data.empty:
                                plot_pred_data = pd.concat([history_data.iloc[[-1]], future_data], ignore_index=True)
                                fig_pred_trend.add_trace(go.Scatter(x=plot_pred_data['Year'], y=plot_pred_data['Burden Mean'], name='Kematian (Prediksi)', line=dict(color='#636EFA', dash='dash'), mode='lines+markers'))
                                fig_pred_trend.add_trace(go.Scatter(x=plot_pred_data['Year'], y=plot_pred_data['Exposure Mean'], name='Polusi (Prediksi)', line=dict(color='#00CC96', dash='dash'), yaxis='y2', mode='lines+markers'))
                            fig_pred_trend.add_vline(x=last_data_year + 0.5, line_width=2, line_dash="dash", line_color="grey")
                            fig_pred_trend.update_layout(title=f"Tren & Prediksi di {pred_country_to_use}", xaxis_title="Tahun",
                                                        yaxis=dict(title="Kematian", title_font=dict(color="#636EFA"), tickfont=dict(color="#636EFA")),
                                                        yaxis2=dict(title="Polusi", title_font=dict(color="#00CC96"), tickfont=dict(color="#00CC96"), overlaying='y', side='right'),
                                                        legend=dict(x=0.1, y=1.1, orientation="h"))
                            st.plotly_chart(fig_pred_trend, use_container_width=True)
            except Exception as e: st.exception(e)
        else: st.info("Masukkan parameter prediksi di sebelah kiri dan klik 'Jalankan Prediksi'.")

with tab4:
    st.markdown("### Filter Data Mentah")
    col_filter_t4_1, col_filter_t4_2 = st.columns(2)
    region_col_name_t4 = filter_data['region_column']
    
    with col_filter_t4_1:
        regions_t4 = ["Global"] + filter_data['regions']
        selected_region_t4 = st.selectbox("Pilih Region:", regions_t4, key="tab4_global_region")
    with col_filter_t4_2:
        countries_list_t4 = ["Semua Negara"]
        if selected_region_t4 == "Global": countries_list_t4 += filter_data['countries']
        elif region_col_name_t4 in df.columns: countries_list_t4 += sorted(df[df[region_col_name_t4] == selected_region_t4]['Country'].unique().tolist())
        selected_country_t4 = st.selectbox("Pilih Negara:", countries_list_t4, key="tab4_global_country")
    
    df_filtered_t4, display_name_t4 = df, "Global"
    try:
        if selected_country_t4 != "Semua Negara":
            df_filtered_t4 = df[df['Country'] == selected_country_t4]
            display_name_t4 = selected_country_t4
        elif selected_region_t4 != "Global" and region_col_name_t4 in df.columns:
            df_filtered_t4 = df[df[region_col_name_t4] == selected_region_t4]
            display_name_t4 = f"Region: {selected_region_t4}"
    except Exception as e: st.error(f"Error filter: {e}"); df_filtered_t4 = df 
    st.markdown("---")

    st.header(f"Data Mentah untuk: {display_name_t4}")
    if not df_filtered_t4.empty:
        st.write(f"Menampilkan {len(df_filtered_t4)} baris data yang telah difilter.")
        st.dataframe(df_filtered_t4)
        st.download_button(
            label="Download Data (CSV)",
            data=df_filtered_t4.to_csv(index=False).encode('utf-8'),
            file_name=f'filtered_data_{display_name_t4.replace(":", "_")}.csv',
            mime='text/csv',
        )
    else:
        st.warning("Tidak ada data untuk ditampilkan berdasarkan filter saat ini.")