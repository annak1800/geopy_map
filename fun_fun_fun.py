######### annas personal fun functions #########

# ## import libraries
# import os
# import numpy as np
# import pandas as pd
# import geopandas as gpd
# from osgeo import gdal, ogr, osr

import ee # type: ignore
import folium # type: ignore
from IPython.display import display

# clip image to geometry

def clip_to_geometry(image, geometry):
    return image.clip(geometry)

# old scaling function (currently not used)

def ScaleMask_S2(img):
    ## Scale the bands
    refl = img.select(['B2', 'B3', 'B4', 'B8', 'B8A', 'B11', 'B12']).multiply(0.0001)
    img_SR = ee.Image(refl).addBands(img.select(['QA60']))
    
    ## mask cloud
    # Get the QA band and apply the bits
    qa = ee.Image(img_SR).select(['QA60'])
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    
    # Both flags should be set to zero, indicating clear conditions
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    
    img_SR_cl = ee.Image(img_SR).select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']).updateMask(mask)
    return img_SR_cl

# reproject image

def reproject_to_3035(img):
    return img.reproject(crs='EPSG:3035', scale=10)

########### cloud masking with sen2clodless ################

# Define the cloud masking functions
def get_s2_sr_cld_col(aoi, start_date, end_date):
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 60)))

    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi)
        .filterDate(start_date, end_date))

    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))

def add_cloud_bands(img):
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')
    is_cloud = cld_prb.gt(50).rename('clouds')
    return img.addBands(ee.Image([cld_prb, is_cloud]))

def add_shadow_bands(img):
    not_water = img.select('SCL').neq(6)
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(0.15*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, 1*10)
        .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
        .select('distance')
        .mask()
        .rename('cloud_transform'))
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

def add_cld_shdw_mask(img):
    img_cloud = add_cloud_bands(img)
    img_cloud_shadow = add_shadow_bands(img_cloud)
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(50*2/20)
        .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
        .rename('cloudmask'))
    return img_cloud_shadow.addBands(is_cld_shdw)

def apply_cld_shdw_mask(img):
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select('cloudmask').Not()

    # Subset reflectance bands and update their masks, return the result.

    return img.select(['B2', 'B3', 'B4', 'B8', 'B8A', 'B11', 'B12']).updateMask(not_cld_shdw)

############# cloud masking ended ################


# Define a method for displaying Earth Engine image tiles to a folium map.
def add_ee_layer(self, ee_image_object, vis_params, name, show=True, opacity=1, min_zoom=0):
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name=name,
        show=show,
        opacity=opacity,
        min_zoom=min_zoom,
        overlay=True,
        control=True
        ).add_to(self)

# calculate indices EVI and NDMI
def add_indices(img):
    ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
    evi = img.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
            'NIR': img.select('B8'),
            'RED': img.select('B4'),
            'BLUE': img.select('B2')
    }).rename('EVI')
    ndmi = img.normalizedDifference(['B8', 'B11']).rename('NDMI')
    return img.addBands(ndvi).addBands(evi).addBands(ndmi)

# get monthly STMs
# Define reducers
mean = ee.Reducer.mean().unweighted()
sd = ee.Reducer.stdDev().unweighted()
percentiles = ee.Reducer.percentile([10, 25, 50, 75, 90]).unweighted()
allMetrics = mean.combine(sd, sharedInputs=True).combine(percentiles, sharedInputs=True)

# monthly STMs
def monthly_STM(img_col):
    def apply_reducers(month):
        start_date = ee.Date.fromYMD(2021, month, 1)
        end_date = start_date.advance(1, 'month')
        monthly_col = img_col.filterDate(start_date, end_date)
        stm = monthly_col.reduce(allMetrics)
        return stm.set('month', month)
    
    months = ee.List.sequence(1, 12)
    monthly_stms = months.map(apply_reducers)
    return ee.ImageCollection.fromImages(monthly_stms)

# seasonal STMs (Spring and Summer)
def seasonal_STM(img_col):
    def apply_reducers(start_month, end_month):
        start_date = ee.Date.fromYMD(2021, start_month, 1)
        end_date = ee.Date.fromYMD(2021, end_month, 30)
        seasonal_col = img_col.filterDate(start_date, end_date)
        stm = seasonal_col.reduce(allMetrics)
        return stm.set('start_month', start_month).set('end_month', end_month)
    
    april_june_stm = apply_reducers(4, 6)
    july_september_stm = apply_reducers(7, 9)
    
    return ee.ImageCollection.fromImages([april_june_stm, july_september_stm])

# Add the Earth Engine layer method to folium.
folium.Map.add_ee_layer = add_ee_layer

# Function to display a specific month's STM
def display_monthly_stm(monthly_stms, month, aoi):
    # Select the image for the specified month
    monthly_img = ee.Image(monthly_stms.filter(ee.Filter.eq('month', month)).first()).multiply(0.0001)
    
    # Define visualization parameters
    vis_params = {
        'bands': ['B4_mean', 'B3_mean', 'B2_mean'],  # Adjust bands as needed
        'min': 0,
        'max': 0.5,
        'gamma': 1.4
    }
    
    # Create a folium map object centered on the AOI
    center = aoi.centroid(10).coordinates().reverse().getInfo()
    m = folium.Map(location=center, zoom_start=12)
    
    # Add the monthly STM layer to the map
    m.add_ee_layer(monthly_img, vis_params, f'Monthly STM - Month {month}')
    
    # Add a layer control panel to the map
    m.add_child(folium.LayerControl())
    
    # Display the map
    display(m)

# Get Sentinel-1 collection
def get_s1_data(aoi, start_date, end_date):
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')) \
        .select('VV', 'VH')
    return s1

# Function to display a specific month's STM
def display_monthly_stm_s1(monthly_stms, month, aoi):
    # Select the image for the specified month
    monthly_img = ee.Image(monthly_stms.filter(ee.Filter.eq('month', month)).first())
    
    # Define visualization parameters
    vis_params = {
        'bands': ['VV_mean', 'VH_mean', 'VV_mean'],  # Adjust bands as needed
        'min': -25,
        'max': 5,
        'gamma': 1.4
    }
    
    # Create a folium map object centered on the AOI
    center = aoi.centroid(10).coordinates().reverse().getInfo()
    m = folium.Map(location=center, zoom_start=12)
    
    # Add the monthly STM layer to the map
    m.add_ee_layer(monthly_img, vis_params, f'Monthly STM - Month {month}')
    
    # Add a layer control panel to the map
    m.add_child(folium.LayerControl())
    
    # Display the map
    display(m)


# synthmix function
import pandas as pd
import numpy as np

def synthmix(df, cl_target, cl_background, n_samples=1000,
             mix_complexity=[2, 3, 4], p_mix_complexity=[0.7, 0.2, 0.1],
             within_class_mixture=True, include_endmember=True):
    """
    Function to generate synthetic training data mixtures from pure endmember spectra.

    df:                 (pd.DataFrame) Input dataframe. First column must contain the
                        class-IDs in integer format. Remaining columns must 
                        contain the features to be mixed. 
    cl_target:          (int) Target class' integer ID value.
    cl_background:      (list) Background class' integer ID value(s). List for 
                        multiple classes, e.g. [2, 3, 4].
    n_samples:          (int) Number of synthetic training points to generate.
    mix_complexity:     (list) List with desired number of possible mixtures
                        between different classes.
    p_mix_complexity:   (list) List containing desired occurrence probabilities 
                        associated to the number of possible mixtures 
                        (i.e. mix_complexity). Must be of same length as 
                        'mix_complexity' argument.
    
    returns:            (pd.DataFrame) Dataframe with linearly mixed features and 
                        corresponding fraction of target class (i.e. cl_target)
    """
    
    # Total number of classes
    all_ems = [cl_target] + cl_background
    n_em = len(all_ems)
    
    # Create empty dataframe to store training data
    df_mixture = pd.DataFrame(columns=list(df.columns[1:]) + ['fraction'])
    
    # Index list of EMs for sampling
    idx_em = {em: df.index[df.iloc[:, 0] == em].tolist() for em in all_ems}
    
    # Vector for fraction calculation
    zero_one = np.zeros(len(df))
    zero_one[idx_em[cl_target]] = 1
    
    # Iterator for generating each synthetic mixture
    for i in range(n_samples):
        
        if len(p_mix_complexity) == 1:
            complexity = mix_complexity[0]
        else:
            # Sample mixing complexity based on mixing likelihoods
            complexity = np.random.choice(mix_complexity, p=p_mix_complexity)
        
        # Select background EMs which will be included in the mixture
        if within_class_mixture:
            background = np.random.choice(all_ems, complexity - 1, replace=True)
        else:
            background = np.random.choice(cl_background, complexity - 1, replace=False)
        
        # Sample indices of selected EMs
        response = [cl_target] + list(background)
        drawn_index = [np.random.choice(idx_em[r]) for r in response]
        drawn_features = df.iloc[drawn_index, 1:]
        drawn_fraction = zero_one[drawn_index]
        
        # Sample random weights
        drawn_weights = []
        for j in range(complexity - 1):
            if j == 0:
                weight = np.random.uniform()
            else:
                weight = np.random.uniform() * (1. - sum(drawn_weights))
            drawn_weights.append(weight)
        drawn_weights.append(1. - sum(drawn_weights))
        
        # Calculate mixtures and associated fractions
        calc_mixtures = np.sum(drawn_features.values * np.array(drawn_weights)[:, None], axis=0)
        calc_fraction = np.sum(drawn_fraction * drawn_weights)
        
        # Append to dataframe
        df_mixture.loc[len(df_mixture)] = list(calc_mixtures) + [calc_fraction]
    
    if include_endmember:
        df_endmember = df.iloc[:, 1:].copy()
        df_endmember['fraction'] = zero_one
        df_mixture = pd.concat([df_mixture, df_endmember], ignore_index=True)
    
    return df_mixture

