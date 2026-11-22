# Ayara: Inflation Analytics and Forecasting

## Implemented application, experimental receipt signals and a research direction for local price measurement

**Anand Indukuri**  
Computer Science and Data Science student, University of Wisconsin Madison  
Public technical edition | September 2026

Live application: [ayara1.streamlit.app](https://ayara1.streamlit.app/)

### Abstract

Ayara is a Python and Streamlit application that brings Indian Consumer Price Index data, time series forecasts, multilingual explanations and receipt extraction into one interface. Its current analytical scope is five spending categories across national rural, urban and combined sectors. I trained and integrated 15 forecasting models, one for each category and sector combination. The application loads those previously trained models to generate forecasts and model supplied intervals.

A separate receipt workflow uses Gemini to extract structured information from images and PDFs, assigns items to categories through local text rules and offers CSV export and optional contribution to Google Sheets. An experimental layer compares receipt price averages across two periods and applies a capped adjustment beside the original forecast. This layer does not retrain the models or constitute a geographically resolved inflation index.

This edition documents the implemented system, preserves the original project's reported model results with their limitations and defines the work required for a credible local price index. Ayara began during my Data Scientist Internship at Crimson Education from July to December 2025 and continued as an independent project afterward. The methods and results described here combine that research history with the current application implementation. [1][2]

## 1. Purpose and present scope

Ayara explores how historical inflation data and forecasting can be made easier to inspect and discuss. A visitor can select a category and sector, examine the historical series, request a future month and read an explanation in English, Hindi or Telugu. The project emphasizes access to the underlying series and understandable presentation rather than a single headline measure.

The application covers Fruits, Vegetables, Spices, Fuel and light and Health. Each category has Rural, Urban and Combined series. These are national sector distinctions. They do not identify an individual neighborhood, household or merchant's local inflation rate.

The project includes implemented and proposed capabilities:

| Layer | Present status | What it supports |
| --- | --- | --- |
| CPI analysis and forecasting | Implemented | Historical charts, category and sector selection, model forecasts and model generated intervals |
| Receipt extraction and contribution | Implemented, with external service configuration | Structured extraction, local item categorization, CSV export and optional shared data contribution |
| Receipt informed forecast adjustment | Implemented as an experiment | A bounded comparison beside the original forecast, based on receipt price averages |
| Geographic inflation index | Proposed research | Local sampling, comparable item prices, explicit basket construction and validation remain to be developed |

Receipts provide observations of particular purchases. Turning them into an index requires decisions about which products are comparable, which locations and households are represented and how prices are weighted through time. The current project does not claim to have resolved those measurement problems.

## 2. Data and application architecture

### 2.1 Historical data and preparation

The original project paper documents a monthly CPI dataset sourced through the Reserve Bank of India's Database on Indian Economy and attributed to the National Statistical Office. It describes coverage from January 2013 through July 2025 on a base of 2012 equal to 100. This is the documented research dataset, not a claim that the application automatically retrieves the latest official release. [1][3][4]

The reported research preparation included date parsing, ordering and deduplication, standardizing product columns, reindexing to a monthly calendar and interpolating missing values. These steps belong to dataset preparation. The current application reads a prepared CSV and normalizes dates for display and forecasting; it does not contain the full cleaning or training pipeline.

For a future reproducible evaluation, preparation must be recorded together with the data version and evaluation splits. In particular, any treatment of missing values should be checked to ensure that evaluation periods do not contribute information to earlier training periods. This edition does not reproduce the original preprocessing runs or establish whether every such condition was satisfied.

### 2.2 Components and data flow

The implementation separates historical data, saved forecasting models and receipt processing. [2]

1. **Historical analysis:** A prepared CSV supplies monthly category and sector series. The interface filters the selection and draws historical charts.
2. **Forecast inference:** A category and sector mapping selects a serialized model. The application calculates the requested horizon and retrieves predicted means and available model intervals.
3. **Explanations:** The Analysis view includes written commentary for category and audience selections. Gemini can translate or simplify that text. The forecasting view also computes descriptive statistics and offers an optional AI explanation.
4. **Receipt processing:** An image or PDF is sent to Gemini for structured extraction. Local rules categorize the returned items. The user can export the table or explicitly save categorized rows to a configured Google Sheet.
5. **Experimental adjustment:** Category level receipt averages generate a limited overlay on the saved model's forecast. The original model output remains separately visible.

The application depends on a prepared dataset, saved model artifacts and its software environment. AI features additionally require Gemini configuration, while shared contributions require Google Sheets credentials and access. A functioning forecast workflow is evidence of application inference, not a reproduction of the model training procedure.

## 3. Forecasting method and reported evaluation

### 3.1 Training and inference

I trained 15 seasonal ARIMA models for the category and sector combinations. The original methodology describes seasonal decomposition, Augmented Dickey Fuller testing, inspection of autocorrelation and partial autocorrelation and a grid search evaluated through time series cross validation. It reports selecting specifications using mean validation error, followed by evaluation on a final holdout of twelve months. [1]

Seasonal ARIMA is written here as `(p,d,q)(P,D,Q)` with a seasonal period of twelve months. The first group specifies the nonseasonal component; the second specifies the seasonal component. The model orders and error values below are retained from the original report.

The deployed interface performs inference with serialized models. For a requested date before the observed record, it reports that the date is outside the available data. Within the observed record, it returns an observed value when present. For a future month, it calculates the forecast horizon from the final observed month and invokes the selected model. Training, selection and evaluation are separate from this request path. [2]

### 3.2 Reported model results

**Reported results from the original project paper.** The table reproduces its model specifications and final twelve month holdout RMSE values in CPI index points. The training and evaluation runs have not been independently reproduced for this edition. The original report describes the procedure but does not provide the complete search grid, fold definitions, evaluation outputs or executable training pipeline within the paper. [1]

| Category | Sector | SARIMA order | Reported RMSE |
| --- | --- | --- | ---: |
| Fruits | Rural | (2,0,4)(0,4,2) | 7.31 |
| Fruits | Urban | (2,3,4)(0,0,0) | 5.73 |
| Fruits | Combined | (2,3,4)(0,1,1) | 10.99 |
| Vegetables | Rural | (2,1,1)(1,1,1) | 27.27 |
| Vegetables | Urban | (1,1,2)(4,0,1) | 34.41 |
| Vegetables | Combined | (1,1,2)(4,0,1) | 28.30 |
| Spices | Rural | (0,2,1)(0,1,1) | 0.97 |
| Spices | Urban | (0,2,1)(0,1,1) | 1.78 |
| Spices | Combined | (0,2,1)(0,1,1) | 1.31 |
| Fuel and light | Rural | (0,1,0)(2,1,1) | 1.30 |
| Fuel and light | Urban | (0,1,0)(0,1,1) | 1.10 |
| Fuel and light | Combined | (0,1,0)(2,1,1) | 0.98 |
| Health | Rural | (1,1,2)(0,1,1) | 0.40 |
| Health | Urban | (0,2,1)(0,1,1) | 0.30 |
| Health | Combined | (1,1,2)(0,1,1) | 0.55 |

RMSE summarizes error across the evaluated observations. A value below two index points does not mean that every forecast was within two points and it does not guarantee performance on future dates. These results apply to individual series and the reported evaluation period; they should not be collapsed into a single accuracy percentage for the application.

The original report shows higher errors for Vegetables than for the smoother series in its evaluation. That observation motivates further investigation. But it does not by itself establish which additional model or input would improve future forecasts. A subsequent comparison should evaluate all candidates on consistent data and temporal splits.

### 3.3 Uncertainty and limits

The application requests and plots intervals supplied by the forecasting model. This edition does not claim measured coverage or calibrated uncertainty. The receipt adjusted forecast has no separately estimated interval: displaying the original model's interval does not validate uncertainty around the adjustment.

The 15 models are separate category and sector forecasts. The current implementation does not enforce consistency between Rural, Urban and Combined predictions, retrain models automatically or establish that every saved model's training endpoint matches every later dataset version. Those are appropriate targets for a documented model release and evaluation process.

## 4. Receipt extraction and the experimental price signal

### 4.1 Extraction and contribution

The current receipt workflow uses Gemini to request merchant, transaction date, currency, total, tax and line item fields. It accepts images and PDFs, parses the returned JSON and displays structured rows. Images are reencoded when the image processing library is available; PDFs are passed to the service as document bytes. Reencoding does not redact sensitive text visible on a receipt. [2]

Item categorization is performed locally through keyword matching and regular expressions. The categories match those used in CPI analysis, with unmatched descriptions assigned to Other. The application includes basic date and numeric cleaning, CSV export and an explicit option to contribute rows to Google Sheets. Stored fields include dates, item descriptions, quantities, prices, currency, merchant, category and a logging timestamp.

The extraction result is not an authenticated transaction record. The current code does not fully validate every returned field, reconcile item totals, measure extraction accuracy or normalize units, brands and package sizes. A real receipt can still be incomplete, misread or unsuitable for comparing prices across purchases.

Tesseract was part of my earlier internship receipt work. Gemini describes the current implementation documented in this edition. This distinction preserves the project's development history without implying that both engines form one current extraction pipeline.

### 4.2 How the adjustment is calculated

The application reads contributed rows for a selected category and compares two adjacent windows of 60 days, anchored to the newest receipt date in that category. The anchor is the newest observation, not necessarily today's date. Each window's mean uses available numeric unit prices. [2]

The relative change is calculated as:

`raw_change_percent = 100 × (recent_mean / previous_mean - 1)`

For a nonzero adjustment, the implementation requires at least five recent item rows and an absolute change of at least one percent. It caps the raw change at positive or negative five percent and applies a weight that reaches one at twenty recent item rows:

`weight = min(1, recent_row_count / 20)`

`adjustment_percent = clip(raw_change_percent, -5, 5) × weight`

`adjusted_forecast = original_forecast × (1 + adjustment_percent / 100)`

These constants are implementation choices, not statistically validated thresholds. The count is item rows, not unique receipts, independent transactions or contributors. The adjustment multiplies the future forecast by a constant factor; it does not estimate a new time series model. The original prediction remains available for comparison.

### 4.3 Why the signal remains experimental

Category averages can change because the mix of purchased products changes, even when the price of each product is unchanged. The current calculation does not construct a matched basket, standardize package quantities, convert currencies, resolve geography or filter anomalous prices. It also does not distinguish rural from urban receipt contributions when applying a category signal.

The row count weight is not statistical confidence. Rows with missing numeric prices can still contribute to the count and the previous window has no comparable minimum count requirement. These limitations prevent a claim that the overlay estimates a local inflation rate or improves predictive accuracy. Its value at this stage is as an implemented experiment whose behavior can be inspected and evaluated.

## 5. Explanations, reliability and data handling

### 5.1 What the AI layer does

The Analysis view translates or simplifies a library of written commentary. Ordinary English commentary can be shown without a model call; requesting simplification, including in English, can invoke Gemini. The forecasting view derives summary statistics from observed values and forecasts and can request an additional explanation. [2]

These functions support presentation. Translation and generation may alter meaning or introduce errors and their quality has not been established through a published language evaluation. The interface should therefore retain the underlying data and distinguish generated explanation from observed values and model output. The current functionality does not constitute a personalized household adviser or retrieval system over a user's private spending history.

### 5.2 Implemented controls and their boundaries

The source reads API credentials from configuration rather than embedding a key. It includes upload consent, file size and count limits, request counting, caching and retries for some transient provider failures. Saving receipt rows to the shared sheet is a separate user action. These are implemented controls, not a security certification or a guarantee of availability. [2]

The quota logic runs in the application host. It reserves capacity before a logical request that can include retries. So it should not be described as a strict provider billing limit or a control running solely on the visitor's device. Deployment behavior and provider availability remain external dependencies.

Uploaded receipt contents are sent to Gemini for extraction. The extraction function uses caching without an explicit expiration or session close purge in the reviewed source. Optional contributions are persisted in Google Sheets. Consequently, this edition does not promise that closing a browser session deletes all receipt information or that nothing is stored by the application.

Formal anonymization, automatic redaction of personal information, enforceable retention controls and independently assessed privacy protections are not established by the current implementation. A future public data release must address these before representing contributed records as anonymous or safe for unrestricted reuse. The application is an educational project and its forecasts are estimates rather than guaranteed outcomes or individualized financial advice.

## 6. Research direction and next validation work

The proposed local index is a separate research objective. Alternative price measurement provides relevant precedent. But it does not validate Ayara's present receipt signal. The Billion Prices Project is one documented example of studying online price data for measurement and research. [5]

The next work should establish evidence in the following order:

1. **Make the existing forecasting study reproducible.** Package the data version, preparation procedure, model selection settings, temporal splits, saved models and evaluation outputs. Reproduce the reported table before adding new performance claims. Compare simple baselines and candidate models under the same evaluation conditions.
2. **Evaluate extraction and normalization.** Build a documented receipt evaluation set with appropriate permission, assess field correctness and separate extraction failures from categorization failures. Define comparable products, units, currencies, brands and package sizes before aggregating prices.
3. **Define geographic and sampling scope.** Establish how contributions map to locations and which populations and stores they represent. Decide how a local basket is constructed, weighted and updated. Receipt counts alone cannot establish representative coverage.
4. **Test the receipt overlay as a forecast intervention.** Compare forecasts with and without the adjustment on subsequent observations using a fixed evaluation procedure. Examine whether any result persists across categories and periods. Until then, retain the original forecast and label the overlay experimental.
5. **Measure interval coverage and coherence.** Evaluate uncertainty as well as point forecasts and investigate reconciliation across related sectors. More complex models, external regressors or ensembles should be judged through these comparisons rather than presumed to improve the system.
6. **Align privacy controls with public claims.** Specify retention and deletion behavior, minimize stored fields, redact sensitive information where appropriate and review access to shared data. Publish a data handling description that matches the deployed behavior.

Ayara's demonstrated contribution is the integration of forecasting, receipt extraction, data contribution and accessible explanation in a usable application. Its next research challenge is to establish whether carefully normalized and geographically defined receipt data can add reliable information beyond the existing CPI models.

## References and evidence notes

**[1]** Indukuri, Anand Varma. *Ayara*. Original project white paper, June 2026. The source of the documented research dataset period, training procedure and reported model table reproduced in this edition. The original is an author prepared project report, not a peer reviewed publication.

**[2]** Indukuri, Anand. *Ayara application source*. Python and Streamlit implementation reviewed for this edition, September 2026. Supports the descriptions of current data loading, model inference, Gemini extraction, item categorization, optional Sheets contribution, receipt adjustment and data handling controls. [Live application](https://ayara1.streamlit.app/).

**[3]** Ministry of Statistics and Programme Implementation, National Statistical Office. *Consumer Price Index, base 2012 = 100: monthly releases and methodology*. The official statistical source identified in the original research report. [MOSPI](https://www.mospi.gov.in/).

**[4]** Reserve Bank of India. *Database on Indian Economy: Prices and Wages, Consumer Price Index, Rural, Urban and Combined*. The dataset access source identified in the original research report. [DBIE portal](https://data.rbi.org.in/).

**[5]** Cavallo, Alberto and Roberto Rigobon. 2016. “The Billion Prices Project: Using Online Prices for Measurement and Research.” *Journal of Economic Perspectives*, 30(2), pages 151 to 174. Bibliographic reference retained from the original report as context for alternative price measurement, without asserting that its results establish Ayara's performance.

The model results in Section 3 are reported project results, not new measurements made for this public edition. No usage, extraction accuracy, forecast improvement, geographic coverage or privacy certification figures have been added. Current implementation, historical research and proposed capabilities are distinguished throughout.
