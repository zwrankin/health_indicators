health_indicators
==============================

Exploring national health indicators using primary data sources (the Demographic and Health Survey API) and global estimates (the Global Burden of Disease)

This repo supports the article on [Data, Science, and Sustainable Development](https://medium.com/@zwrankin/data-science-and-sustainable-development-challenging-historical-paradigms-with-k-means-b1b39305e3e7)
and corresponding interactive visualization on Heroku: https://sdg-3.herokuapp.com/
![](readme_images/app_capture.GIF)

### Data sources
There are four data sources queried by `src/data`
- `download_DHS` - Queries the [DHS Program API](http://api.dhsprogram.com/#/index.html), which has household survey data from dozens of countries
- `download_GBD` - Downloads data from the Institute of Health Metrics ([IHME](http://www.healthdata.org/)) using internal tools, though this data is also available via the public API
- `process_SDG` - Processes data downloaded from IHME's [Sustainable Development Goal estimates](http://ghdx.healthdata.org/record/global-burden-disease-study-2017-gbd-2017-health-related-sustainable-development-goals-sdg)
- `query_GBD_api` - Queries the IHME API, which requires a private key (can be requested [here](http://ghdx.healthdata.org/contact))

The final visualization only uses data downloaded from IHME's internal tools. 
