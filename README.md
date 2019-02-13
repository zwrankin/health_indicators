health_indicators
==============================

Exploring national health indicators using primary data sources (the Demographic and Health Survey API) and global estimates (the Global Burden of Disease)

## Visualization
Interactive visualization for clustering of child health indicators: 
https://sdg-indicators.herokuapp.com/
![Alt text](readme_images/app_snapsnot.JPG?raw=true "App snapshot")

### Data sources
There are four data sources queried by `src/data`
- `download_DHS` - Queries the [DHS Program API](http://api.dhsprogram.com/#/index.html), which has household survey data from dozens of countries
- `download_GBD` - Downloads data from the Institute of Health Metrics ([IHME](http://www.healthdata.org/)) using internal tools, though this data is also available via the public API
- `process_SDG` - Processes data downloaded from IHME's [Sustainable Development Goal estimates](http://ghdx.healthdata.org/record/global-burden-disease-study-2017-gbd-2017-health-related-sustainable-development-goals-sdg)
- `query_GBD_api` - Queries the IHME API, which requires a private key (can be requested [here](http://ghdx.healthdata.org/contact))

The final visualization only uses data downloaded from IHME's internal tools. 
