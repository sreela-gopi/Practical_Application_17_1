# **Comparing Classifiers**

###🎯 **Project Goal:**
The goal of this project is to build Machine Learning Classification models that can predict if the client will subscribe a term deposit and compare the performance of different classifiers used.

###🔎 **Problem Context:**
Utilizing a dataset related to marketing bank products over the telephone, the task is to compare the performance of the classifiers we encountered in this section, namely K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines.

##📖 **Notebook & Data**

 The Jupyter Notebook containing the analysis of the problem can be found here: [Notebook](https://github.com/sreela-gopi/Practical_Application_17_1/blob/main/practical_application_assignment_17.1.ipynb)<br>
 The dataset provided for the analysis can be found here : [Dataset](https://github.com/sreela-gopi/Practical_Application_17_1/blob/main/data/bank-additional-full.csv)

##📈 **Findings**
Below are the findings from the analysis and implementation of this project. <br>
**Findings from Exploratory Data Analysis(EDA)** <br>

**Univariate Analysis:**
1. **Age**: Majority of the customers targeted in the marketing campaign are in the middle-aged range (30s-50s).

2. **Duration**: The duration of the last contact has a large concentration of very short calls and a fewer very long calls.

3. **Campaign**: The number of contacts during this campaign for a client shows that most clients were contacted only a few times (1-2 times)

4. **Pdays**: This variable indicates the number of days since the client was last contacted from a previous campaign. The histogram for 
pdays shows a very large bar at the value of 999. This value signifies that the client was not previously contacted, indicating that a significant portion of the dataset consists of new or previously untouched contacts.

5. **Previous**: The previous variable, which counts the number of contacts before this campaign, would show a distribution where a very large number of clients have zero previous contacts. This reinforces the finding from the pdays variable that many of the contacts were part of the first-time campaign.

**Distribution of economic indices**:<br>

6.  **emp.var.rate (Employment Variation Rate)**: Employment Variation Rate changed in large discrete steps rather than a smooth, continuous variation over the period. This shows major economic trends.

7.  **cons.price.idx (Consumer Price Index)**: Consumer Price Index distribution shows clusters of values rather than a normal distribution which suggests that consumer prices were relatively stable or changed in specific periods.

6.  **cons.conf.idx (Consumer Confidence Index)**: Consumer Confidence Index shows a distribution of cunsumer sentiments during the time of marketing calls,  with clear highs and lows at specific values.

9. **euribor3m (Euribor 3 month rate)**: Euribor 3 month rate shows that interest rates changed only a few times throughout the campaign period.

10. **nr.employed (No.of Employees)** : No.of Employees is an economic indicator, it reflects the current state of the labor market and  economic health. The histogram shows a discrete distribution with distinct peaks at a few specific values which indicates that the number of employees was reported at a few fixed points in time, rather than changing continuously.
11. **Job**:The most common job is 'admin.', followed by 'blue-collar' and 'technician'. The least represented jobs are 'housemaid', 'unemployed', 'students' and 'unknown'. This indicates that the bank's marketing campaign primarily targets a few specific professions.
12. **Marital Status**: The marital status plot shows that the majority of clients are 'married', followed by 'single', and then 'divorced'. The number of clients with 'unknown' marital status is very small. This suggests the campaign was mainly focused on a married demographic.
13. **Education**: The education plot shows a significant number of clients with a 'university.degree' or 'high.school' education, followed by those with 'basic.9y' or 'professional.course'.
14. **Default (Financial Status)**: The count plot shows a very high number of 'no' responses which indicates that a vast majority of customers have no credit in default.
15. **Housing**: Most customers have a housing loan.
16. **Loan**: Most customers don't have a personal loan.
17. **Contact**: Shows a significant preference for contacting clients via 'cellular' phone over 'telephone'.
18. **Month** : Plot of the last contact month shows a very high concentration of calls in the months of May, July, and August, and very few in December or March. This suggests a seasonal or targeted campaign schedule.
19. **Day of Week**: The calls are relatively evenly distributed across the days of the week, from Monday to Friday
20. **Poutcome**: This plot provides insights into the success of the previous campaign. It shows a high number of 'nonexistent' outcomes, which means that for the vast majority of clients, there was no previous campaign interaction. The number of successful and failed outcomes is very small in comparison.
21. The count plot for the target variable y (whether the client subscribed to a term deposit) reveals a severe class imbalance. The number of clients who did not subscribe is drastically higher than those who said 'yes'.
22. The high imbalance in the class is a challenge for a predictive model, as it will be much easier to predict the "no" class, and a model's performance should be carefully evaluated using metrics other than just accuracy.

**Bivariate Analysis:**

1. **Strongest Predictors**:
    * **`poutcome` (Outcome of previous campaign)**: The analysis shows that clients who had a successful outcome in a previous campaign are  significantly more likely to subscribe the current one. This suggests that past positive response is a strong indicator of future success.
    * **`contact`**: The plots show that the majority of successful subscriptions came from calls made via cellular phones.
2. **Demographics**:
   * **Job & Education**: While the bank targeted a wide range of jobs, specific groups like 'students' and the 'retired' have a disproportionately higher subscription rate. While university.degree is common, the subscription rate for basic education levels are a little higher.
   * **Marital Status**: Married clients represent the largest group, but the subscription rate for single and divorced clients is higher when considering the total number of people in each group. This suggests that individuals without a spouse might be more attracted to the campaign.
3. **Financial Status**:
   * The analysis shows that clients with no credit in default, those without a personal loan and those with a housing loan, are the most frequent targets of the campaign.
4. **Campaign Timing**:
   * **Month**: The campaign was not evenly distributed throughout the year. Most calls were made in May, Jun, Jul and Aug perhaps due to Bank's strategy. But the subsciption rate is seen higher in the months Oct, Spet, Mar and Dec though the calls made were very less.
   * **Day of week**: The plot suggests a consistent campaign schedule without a significant preference for a specific day.
5. **Strong Positive correlations**:
   * **`duration`**: This has the strongest positive correlation with the target variable `y`. This suggests that the longer the call, the more chances the client is to subscribe. Since the call duration is only known after the outcome, **this variable cannot be used to predict the outcome of a future call**.
   * **`previous`**: The number of contacts made previously to the same client has a positive correlation with target; which means the past campaign efforts make the clients more receptive to the subscription.
6. **Strong Negative correlations**:
   * **`euribor3m`**: There is a strong negative correlation between the interest rate and the subscription outcome. This indicates that clients were less likely to subscribe to a term deposit when the interest rate was high.
   * **`nr.employed`**: This feature also has a strong negative correlation with `y`. This suggests that the campaign was more successful when unemployment was higher.
7. **Weak Correlations**:
   * **`age`**: There is a weak positive correlation between `age` and target variable `y`. This suggests that older individuals, such as retirees, may be more open to this type of subscriptions.
   * **`campaign`**: There is a negative correlation between the number of contacts in the campaign to the target variable. The success rate  decreases as the number of contacts increases.
   * **`pdays`**: The heatmap shows a negative correlation betweeen number of days since previous contact and the taregt variable. But because the high value indicated first time contact, this is misleading.
   * **`emp.var.rate`**: Shows a negative correlation with y, indicating that during periods of a weak economy there were more successful  outcomes for the bank.
   * **`cons.price.idx`**: This also shows a negative correlation with y, indicating that during periods of a weak economy there were more successful  outcomes for the bank.
8. **campaign vs. duration**: This plot shows a very concentrated cluster of data points in the bottom-left corner, representing a high number of clients who were contacted only a few times and had short call durations. Also seen that all successful subscriptions ('yes') are associated with longer-duration calls.

9. **pdays vs. previous**: This plot shows a line of points at pdays around 999. This represents the majority of clients who were not contacted in a previous campaign. The scatter of points in the region where `pdays` < 999 shows a clear negative correlation, indicating that clients contacted more frequently in the past (`previous` > 0) have a shorter time elapsed since their last contact (`pdays` < 999).

**Findings from model evaluation:** <br>

**Interpretation of Tuned Model Performance**:
1. **Decision Tree: Best Test Accuracy and Efficiency**
    * **Test Accuracy (89.46%)**: The Decision Tree achieved the second-highest test accuracy, only slightly behind the SVM. This is a massive improvement over the default Decision Tree model (which had 82.74% test accuracy and severe overfitting).

    * **CV Train Accuracy (89.35%)**: The close proximity of the CV Train and Test Accuracies (89.35% vs. 89.46%) demonstrates that the GridSearchCV process **successfully mitigated the overfitting problem** by finding optimal parameters for max_depth and/or min_samples_leaf.

    * **Total Time**: The tuning process was very fast, making the tuned Decision Tree the most efficient high-performing model.

2. **SVM (SVC): Highest Test Accuracy, Extremely Slow**
    * **Test Accuracy (89.49%)**: The SVM achieved the highest test accuracy, confirming its potential as a powerful classifier.

    * **Total Time**: The execution time is very much higher than all other models. This is a critical factor for deployment, as the gain in accuracy over the Decision Tree and Logistic Regression is likely not worth the massive increase in computational cost.
3. **Logistic Regression: Reliable and Fast**
    * **Test Accuracy (89.37%)**: The accuracy is good, demonstrating a reliable and effective model.

    * **Total Time**: Extremely fast tuning time.

    * **Advantage**: Logistic Regression offers the best interpretability of coefficients, making it easy to explain which factors drive the subscription decision.
4. **K-Nearest Neighbors (KNN): Underperforming**
    * **Test Accuracy (88.98%)**: This is the lowest accuracy among the tuned models and is only about 0.66% better than the baseline (88.32%).

    * **Total Time**: The tuning time is reasonable but the resulting performance is disappointing.

###📌 **Conclusion:** <br>

**Best Model: Decision Tree** 🏆<br>
**Worst Model: KNN** 👎

The tuned Decision Tree is the most practical choice. It provides good accuracy and faster to tune and train than the SVM, and it effectively resolves the overfitting issue found in the default model.



##💡 **Actionable insights to the Business**
1. **Identify the Audience of campaign**:
   * **Age**: The campaign's primary focus was on middle-aged clients (30-50). The bank could develop tailored strategies for younger or older demographics, who might have different financial needs.
   * **Job & Education**: The campaign was focused on 'admin.', 'blue-collar', 'technician', and university-degree clients. The bank should consider creating campaigns to reach other segments, like students or retirees, who may have specific savings or investment needs.
   * **Marital Status**: With a majority of clients being married, the bank could design products or services that appeal to families.

2. **Communication Strategy**:

   * **Contact**: The high number of calls made via 'cellular' phones. The bank should continue to prioritize this channel and potentially explore other digital ways like SMS marketing or banking app notifications.

   * **Campaign Frequency**: Most clients were contacted only once or twice. This suggests that the bank may be missing opportunities for follow-up. The bank could look into a strategy for making follow-ups.

   * **Timing of Campaign**: The concentration of calls in May, July, and November suggests a seasonal marketing push. The bank could analyze the effectiveness of these peak months to determine if the timing is linked to a higher success rate. <br>
   The relatively uniform distribution of calls across weekdays suggests a consistent effort, which is a good practice.

3. **Understand the Customer's Financial Status**:
   * **Loan Status**: A large proportion of clients have a housing loan but not a personal loan. This information can be used to categorize  customers and offer them relevant products.

   * **Previous Outcome**: The high number of 'nonexistent' values in the poutcome column indicates that a large portion of the campaign was directed at new clients or those who were not previously contacted. This suggests an opportunity to create a targeted strategy for clients who were previously contacted but did not subscribe.
4. **Prioritize Past Successes**: The analysis of `poutcome` shows that clients who previously subscribed have a very high chance of subscribing again. The bank should create a specific campaign for this.
5. **Focus on Retirees and Students**: While the general campaign targeted a broad audience, age analysis shows that older clients and students have a disproportionately higher subscription rate. The bank should consider tailoring specific products and marketing for these two demographics.
6. **Don't Give Up After One Failed Attempt**: Clients who were contacted previously but did not subscribe still have a higher subscription rate than those who were never contacted. This suggests that follow-up calls are worthwhile.
7. **Align Campaigns with Economic Conditions**: The analysis of economic indicators like `euribor3m` and `nr.employed` reveals that campaigns were more successful during periods of economic falloff. The bank should plan its major marketing pushes during such periods.
8. **Optimize Campaign Timing by Month**: The line plot of `month` versus `y` can help the bank identify the most effective months for campaigns. By allocating more resources to the months with the highest historical subscription rates, the bank can maximize its profit.
9. **Mobile First**: The bivariate analysis of contact confirms that calls made via cellular phones are much more successful than those made via traditional telephones. The bank should continue to prioritize and optimize its mobile-based communication channels.
10. **Focus on Quality**: The analysis of campaign vs. y shows that the subscription rate drops significantly after the first two contacts. This indicates that excessive contact with a client can be counterproductive. The bank should focus on making a powerful first or second impression rather than overwhelming clients with numerous calls.
11. **Discard `duration` as a Predictor**: Although duration has the highest correlation with subscription, it is not an actionable feature for a predictive model **due to data leakage**. The bank should not use this feature to build its models, as it would lead to a false sense of accuracy.

##🌱 **Next steps & Future Recommentions**
1. **Address Data Imbalance**: This dataset has a low rate of successful deposits (the 'Yes' class). Explore techniques during model training to improve the model's ability to correctly identify the minority class.
2. The priority must be improving the model's ability to find the rare 'yes' cases, which is where the real business value lies.
3. Ensemble techniques such as Random Forest and Gradient Boosting could be employed for improving model performance.

##🛠️ **Tech Stack**

**Python:** Pandas, NumPy, scikit-learn <br>
**Data Visualization:** Matplotlib, Seaborn

This assignment was completed as part of the Practical Application Assignment 17.1 of Professional Certificate in Machine Learning & Artificial Intelligence - 2025
