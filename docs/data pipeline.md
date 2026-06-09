# California house pricing prediction (Data Pipeline)

## 1. Data Understanding

- Downloaded the dataset and saw multiple columns regarding house pricing in california.
	- **Target variable:** `MedHouseValue` (Median House Value)
	- **Other features:**
		1. `MedInc` – Median income in block group
		2. `HouseAge` – Median house age in block group
		3. `AveRooms` – Average number of rooms per household
		4. `AveBedrms` – Average number of bedrooms per household
		5. `Population` – Block group population
		6. `AveOccup` – Average household size
		7. `Latitude` – Block group latitude
		8. `Longitude` – Block group longitude
- Goal is to predict the target variable (house pricing in dollars).
- Feature type: All numerical (`float64`)
- Dataset size: `(20640, 8)`
- Each row is a block group

### First-off obstacles

- Need more clarity on whether geographical coordinates provide any usage in the future.
	- Maybe pricing fluctuates according to similar coordinates in an area
- The feature data are not normalized. When modeling, gradient descent specifically break down without normalization. Gradient updates the weights in the direction of descent. without normalization, `x.T @ errors` would result in huge numerical values and would scale the weight updates drastically. As a result, features on different scales produce gradients on different scales — some weights get huge updates, others tiny ones. The loss surface becomes elongated and gradient descent bounces around instead of converging cleanly.

## Conclusions

- Geographical coordinates play an important role in the price fluctuations. Ocean view houses cost more than inland houses. Same applies for premium locations in California.
- In general, prices fluctuate based on rooms available and house ages too.
- Less population in a block group may indicate more availability of houses to buy, resulting in lower prices. More supply, less price.
- Median income can be a good estimate on what type of people live in that block group. Ex: white collar people, black collar people, business owners, etc.
- Normalize the feature data and fit it into a fixed scale to make gradient descent converge cleanly. Also for data distribution in EDA.

## 2. Data Inspection

- Imported the dataset and displayed first 5 rows. Understood the datatype values of each feature.
- The dataset has 20640 rows as block groups and 8 columns as features. All the 8 features show non-null count, meaning they aren't NaN. No chance for any placeholders for null too since the datatype is float, not object.
- Checked the number of unique values in each feature to see if any categorical features get exposed. None found.
- Performed first-level statistics on the dataset using `df.describe()`. Each feature has different min, max boundaries which means different scaling. Needs to be normalized for better modeling.
	- **IMPORTANT:** Normalizing the data should be done separately for each train-val-test splits.
	- **WHY?**
		1. Anything we learn from the data must be only from the training data, then applied to the testing data. Normalization parameters like mean, std are all learned parameters. We fit the model on them in the training data and then apply it to both validation and testing data.
		2. Test data is supposed to simulate unseen data. If you compute stats on the full dataset before splitting, test data has already leaked information into your normalization. This results in misleading accuracy and doesn't generalize the model to true unseen data.
