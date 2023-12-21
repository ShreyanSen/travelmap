import os
from scipy.io import netcdf
import xarray as xr
import pandas as pd
import numpy as np

import geopandas as gpd
import shapely

class TravelMap:

    def __init__(self, data_dir='./../data/'):

        self.load_data(data_dir)
        self.process_data()
        self.gen_global_grid()
        self.fill_global_grid()
        self.queried_dfs = []

    def load_data(self, data_dir):
        """
        important columns:
            dtr: diurnal temperature (average daily temperature) in celsius
            wet: days with any rain
        """
        datafiles = os.listdir(data_dir)
        df_list = []
        for file in datafiles:
            ds = xr.open_dataset(data_dir + file)
            df = ds.to_dataframe().reset_index().dropna()
            df_list.append(df)
        df_list = [df.drop(['stn'], axis=1) for df in df_list]  # don't need the 'st' column
        self.df_list = df_list

    def process_data(self):
        df_t = self.df_list[0]
        for df in self.df_list[1:]:
            df_t = df_t.merge(df, on=['lon', 'lat', 'time'])

        def cels_to_fahr(x):
            """Americuhhh fuck yeahhh"""
            y = 32 + (x * 1.8)
            return y

        df_t['tmx_f'] = df_t['tmx'].apply(lambda x: cels_to_fahr(x))
        df_t['tmn_f'] = df_t['tmn'].apply(lambda x: cels_to_fahr(x))
        df_t['tmp_f'] = df_t['tmp'].apply(lambda x: cels_to_fahr(x))
        df_t['dtr_f'] = df_t['dtr'].apply(lambda x: cels_to_fahr(x))
        df_t['days_wet'] = df_t['wet'].dt.total_seconds() / (60 * 60 * 24)
        self.global_df = df_t
        # self.global_df.reset_index()
        self.global_df['month'] = self.global_df['time'].dt.month
        self.global_df_monthly = self.global_df.drop(['time'], axis=1).groupby(
            ['lat', 'lon', 'month']).mean().reset_index()

        self.global_monthly_gdf = gpd.GeoDataFrame(
            self.global_df_monthly, geometry=gpd.points_from_xy(self.global_df_monthly.lon, self.global_df_monthly.lat),
            crs="EPSG:4326"
        )

    def gen_global_grid(self, N_BOXES=500):
        # BOXES = 500
        a, b, c, d = (-180.0, -90.0, 180.0, 90.0)  # global bounds
        # TODO: it's not number of boxes! actually the num boxes is (N_BOXES-1)**2. instead need to specify coordinate fineness and
        # write a function that goes from there
        gdf_grid = gpd.GeoDataFrame(
            geometry=[
                shapely.geometry.box(minx, miny, maxx, maxy)
                for minx, maxx in zip(np.linspace(a, c, 361)[:-1], np.linspace(a, c, 361)[1:])
                for miny, maxy in zip(np.linspace(b, d, 181)[:-1], np.linspace(b, d, 181)[1:])
            ],
            crs="epsg:4326",
        )
        self.global_grid_frame = gdf_grid

    def fill_global_grid(self):
        """
        manually verified that on the 1 degree fill this doesn't drop rows in global_monthly_gdf meaning we use all our data when we
        snap it from points to gridded polygons
        """
        self.global_grid_data = gpd.sjoin(self.global_grid_frame, self.global_monthly_gdf, how='left')
        self.active_global_grid_data = self.global_grid_data.loc[:,
                                       ['geometry', 'month', 'tmx', 'tmn', 'tmp', 'tmx_f', 'tmn_f', 'tmp_f',
                                        'days_wet']].dropna()

    def filter_global_grid(self, month_filter=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], tmx_c_filter=[-1000, 1000],
                           tmx_f_filter=[-1000, 1000], tmn_c_filter=[-1000, 1000], tmn_f_filter=[-1000, 1000],
                           tmp_c_filter=[-1000, 1000], tmp_f_filter=[-1000, 1000], days_wet_filter=[-1, 35]):
        """
        defaults set to not filter anything!
        """
        self.filtered_global_grid_data = self.active_global_grid_data.loc[
            (self.active_global_grid_data.tmp_f >= tmp_f_filter[0]) & (
                        self.active_global_grid_data.tmp_f <= tmp_f_filter[1])]
        self.filtered_global_grid_data = self.filtered_global_grid_data.loc[
            (self.filtered_global_grid_data.tmp >= tmp_c_filter[0]) & (
                        self.filtered_global_grid_data.tmp <= tmp_c_filter[1])]
        self.filtered_global_grid_data = self.filtered_global_grid_data.loc[
            (self.filtered_global_grid_data.tmx_f >= tmx_f_filter[0]) & (
                        self.filtered_global_grid_data.tmx_f <= tmx_f_filter[1])]
        self.filtered_global_grid_data = self.filtered_global_grid_data.loc[
            (self.filtered_global_grid_data.tmx >= tmx_c_filter[0]) & (
                        self.filtered_global_grid_data.tmx <= tmx_c_filter[1])]
        self.filtered_global_grid_data = self.filtered_global_grid_data.loc[
            (self.filtered_global_grid_data.tmn_f >= tmn_f_filter[0]) & (
                        self.filtered_global_grid_data.tmn_f <= tmn_f_filter[1])]
        self.filtered_global_grid_data = self.filtered_global_grid_data.loc[
            (self.filtered_global_grid_data.tmn >= tmp_c_filter[0]) & (
                        self.filtered_global_grid_data.tmp <= tmn_c_filter[1])]

        self.filtered_global_grid_data = self.filtered_global_grid_data.loc[
            (self.filtered_global_grid_data.days_wet >= days_wet_filter[0]) & (
                        self.filtered_global_grid_data.days_wet <= days_wet_filter[1])]
        self.filtered_global_grid_data = self.filtered_global_grid_data.loc[
            self.filtered_global_grid_data.month.isin(month_filter)]

    def query_by_lat_lon(self, my_lat=37.77, my_lon=-122.44):
        """
        Queries the global dataframe by lat and lon and then stories the query as well as the results
        """
        # self.queried_df = self.global_df.loc[(np.abs(self.global_df.lon-my_lon)<=0.25) & (np.abs(self.global_df_monthly.lat - my_lat)<=0.25)].groupby('time').agg('mean')

        self.queried_df = self.global_df_monthly.loc[(np.abs(self.global_df_monthly.lon - my_lon) <= 0.25) & (
                    np.abs(self.global_df_monthly.lat - my_lat) <= 0.25)]
        results_dict = {'lat': my_lat, 'lon': my_lon, 'df': self.queried_df}
        self.queried_dfs.append(results_dict)