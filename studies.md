# Self-Study Preparation Guide

**⏳ Estimated Prep Time:** 45–60 minutes

Welcome to our flipped-classroom session, where you'll review foundational concepts beforehand to maximize our time for hands-on coding and debugging. This pre-study focuses on **Time Series Forecasting**, a critical skill for predicting future trends in finance, retail, and operations. By understanding the "shape" of data now, we can dive straight into building predictive models like ARIMA during our live class.

## ⚡ Your Self-Study Tasks

Please complete the following activities before our session.

### 📝 Task 1: Deconstructing Time-Series Data

**Activity:** Read the **"Introduction to Time-Series Data"** and **"Trend, Season, Cycle"** sections in the provided `time_series_forecasting_lesson.ipynb` notebook. Focus on distinguishing between the three core components that make up a signal.

Watch the following explainer videos (~5 mins each) on key topics discussed in this lesson

- [What is Time Series Data](https://youtu.be/FsroWpkUuYI?si=bTI8CS10-iRYuHmQ)
- [What is Time Series Decomposition](https://youtu.be/0ar9extHObg?si=tsuPCrnLpSLIQa_I)
- [What are Exponential Smoothing Models](https://youtu.be/hAD5vVz07ZA?si=ZppkfneYRo2ZqLHS)

**Guiding Questions:**

* How does **Time-Series data** differ fundamentally from the cross-sectional data we have analyzed in previous modules?
* In your own words, what is the specific difference between a **Seasonal** pattern and a **Cyclic** pattern? (Hint: Think about the consistency of the frequency).

### 📝 Task 2: Diagnosing Data Patterns
2
**Activity:** Scroll down to the **"Visualizing Time-Series Data"** and **"Autocorrelation"** sections. Review the code cells and their outputs, specifically the **Seasonal Plots**, **Lag Plots**, and **Autocorrelation (ACF) Plots**.

**Focus your attention on these key visualizations:**

1. **Lag Plots:** How does the shape of the scatter plot indicate if data is random or follows a pattern?
2. **ACF Plot:** Look at the blue shaded region. What does it mean when a bar extends beyond this confidence interval?

### 📝 Task 3: Mental Model Check - Decomposition

**Activity:** Briefly skim the **"Time Series Decomposition"** section at the end of the notebook.

**Guiding Question:**

* The notebook mentions **Additive** () and **Multiplicative** () decomposition. If a retail sales trend is growing exponentially (sales are booming!), which model do you think would fit best?

## 🙌🏻 Active Engagement Strategies

To deepen your retention, try one of the following while you review:

* **Scenario Matching:** Look at the "Examples" section (Housing sales, Treasury bills, etc.). Pick a dataset from your current job or daily life (e.g., website traffic, electricity bill). Which components (Trend, Seasonality, Cycle) does it have?
* **"Code Commentary":** Select the code block for `seasonal_decompose`. Write a brief comment next to it explaining why we might need to separate `trend` from `seasonal` before training a model.
* **Concept Mapping:** Sketch the relationship between *Lag*, *Correlation*, and *Forecasting capability*.

## 📖 Additional Reading Material

* Check out more explainer videos on time series data and forecasting [here](./reference.md#video-playlist).
* [Forecasting: Principles and Practice (Hyndman & Athanasopoulos)](https://otexts.com/fpp3/) - *The gold standard textbook for time series.*


### 🙋🏻‍♂️ See you in the session!
