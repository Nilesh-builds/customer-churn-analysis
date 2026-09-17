# Data Dictionary

| Field | Meaning | Use |
| --- | --- | --- |
| `tenure` | Months with the company | Numeric model feature |
| `MonthlyCharges` | Current monthly charge | Numeric model feature and revenue scenario |
| `TotalCharges` | Charges accumulated to date | Numeric model feature; blank source values become zero |
| `Contract` | Contract commitment | Categorical feature and main business segment |
| `InternetService` | Internet service type | Categorical feature and segment |
| `PaymentMethod` | Payment method | Categorical feature and segment |
| `SeniorCitizen` | Senior-citizen indicator | Categorical/binary feature and subgroup review |
| `Churn` | Whether the customer left | Target: `1` for Yes, `0` for No |

Other service and household fields are retained as candidate categorical
features. `customerID` is removed before modeling because it is an identifier.
