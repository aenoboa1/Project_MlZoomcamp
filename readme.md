In the following midterm project , We will use the [Hotel booking demand data set](https://www.kaggle.com/jessemostipak/hotel-booking-demand) from Kaggle, to predict whenever a customer will cancel his booking request or not .


![alt](https://i2.wp.com/clark.com/wp-content/uploads/2018/07/hotelbooking.jpg?fit=875%2C400&ssl=1)




## Data description 
| Variable                    | Type        | Description                                                                                                                                                                                                                                                                                                |   |
|-----------------------------|-------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---|
| ADR                         | Numeric     | Average Daily Rate                                                                                                                                                                                                                                                                       |   |
| Adults                      | Integer     | Number of adults                                                                                                                                                                                                                                                                                           |   |
| Agent                       | Categorical | ID of the travel agency that made the bookinga                                                                                                                                                                                                                                                             |   |
| ArrivalDateDayOfMonth       | Integer     | Day of the month of the arrival date                                                                                                                                                                                                                                                                       |   |
| ArrivalDateMonth            | Categorical | Month of arrival date with 12 categories: “January” to “December”                                                                                                                                                                                                                                          |   |
| ArrivalDateWeekNumber       | Integer     | Week number of the arrival date                                                                                                                                                                                                                                                                            |   |
| ArrivalDateYear             | Integer     | Year of arrival date                                                                                                                                                                                                                                                                                       |   |
| AssignedRoomType            | Categorical | Code for the type of room assigned to the booking. Sometimes the assigned room type differs from the reserved room type due to hotel operation reasons (e.g. overbooking) or by customer request. Code is presented instead of designation for anonymity reasons                                           |   |
| Babies                      | Integer     | Number of babies                                                                                                                                                                                                                                                                                           |   |
| BookingChanges              | Integer     | Number of changes/amendments made to the booking from the moment the booking was entered on the PMS until the moment of check-in or cancellation                                                                                                                                                           |   |
| Children                    | Integer     | Number of children                                                                                                                                                                                                                                                                                         |   |
| Company                     | Categorical | ID of the company/entity that made the booking or responsible for paying the booking. ID is presented instead of designation for anonymity reasons                                                                                                                                                         |   |
| Country                     | Categorical | Country of origin. Categories are represented in the ISO 3155–3:2013 format [6]                                                                                                                                                                                                                            |   |
| CustomerType                | Categorical | Type of booking, assuming one of four categories:                                                                                                                                                                                                                                                          |   |
| DaysInWaitingList           | Integer     | Number of days the booking was in the waiting list before it was confirmed to the customer                                                                                                                                                                                                                 |   |
| DepositType                 | Categorical | Indication on if the customer made a deposit to guarantee the booking. This variable can assume three categories:  - No Deposit – no deposit was made; - Non Refund – a deposit was made in the value of the total stay cost; - Refundable – a deposit was made with a value under the total cost of stay. |   |
| DistributionChannel         | Categorical | Booking distribution channel. The term “TA” means “Travel Agents” and “TO” means “Tour Operators”                                                                                                                                                                                                          |   |
| IsCanceled                  | Categorical | Value indicating if the booking was canceled (1) or not (0)                                                                                                                                                                                                                                                |   |
| IsRepeatedGuest             | Categorical | Value indicating if the booking name was from a repeated guest (1) or not (0)                                                                                                                                                                                                                              |   |
| LeadTime                    | Integer     | Number of days that elapsed between the entering date of the booking into the PMS and the arrival date                                                                                                                                                                                                     |   |
| MarketSegment               | Categorical | Market segment designation. In categories, the term “TA” means “Travel Agents” and “TO” means “Tour Operators”                                                                                                                                                                                             |   |
| Meal                        | Categorical | Type of meal booked. Categories are presented in standard hospitality meal packages:  Undefined/SC – no meal package; BB – Bed & Breakfast; HB – Half board (breakfast and one other meal – usually dinner); FB – Full board (breakfast, lunch and dinner)                                                 |   |
| PreviousBookingsNotCanceled | Integer     | Number of previous bookings not cancelled by the customer prior to the current booking                                                                                                                                                                                                                     |   |
| PreviousCancellations       | Integer     | Number of previous bookings that were cancelled by the customer prior to the current booking                                                                                                                                                                                                               |   |
| RequiredCardParkingSpaces   | Integer     | Number of car parking spaces required by the customer                                                                                                                                                                                                                                                      |   |
| ReservationStatus           | Categorical | Reservation last status, assuming one of three categories:  Canceled – booking was canceled by the customer;  Check-Out – customer has checked in but already departed;   No-Show – customer did not check-in and did inform the hotel of the reason why                                                   |   |
| ReservationStatusDate       | Date        | Date at which the last status was set. This variable can be used in conjunction with the ReservationStatus to understand when was the booking canceled or when did the customer checked-out of the hotel                                                                                                   |   |
| ReservedRoomType            | Categorical | Code of room type reserved. Code is presented instead of designation for anonymity reasons                                                                                                                                                                                                                 |   |
| StaysInWeekendNights        | Integer     | Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel                                                                                                                                                                                                              |   |
| StaysInWeekNights           | Integer     | Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel                                                                                                                                                                                                                   |   |
| TotalOfSpecialRequests      | Integer     | Number of special requests made by the customer (e.g. twin bed or high floor)                                                                                                                                                                                                                              |   |

Obviously there's a lot of features in this dataset, In this case ,I deleted some features I thought harmed the performance and generality of the model predictions , such as : `Country`, `ReservationStatus`,`ReservationStatusDate`,`AssignedRoomType`, among others. 

With the main one being  `ReservationStatus` , this feature being the clear one that can cause data leakage , and the model may reach 100 % accurate predictions with it .

## Virtual Environment : `Pipenv 2021.5.29` 

<b>Python version: ` Python 3.9.7`🐍  </b>


Versions/requirements used inside the virtual environment:

- ` xgboost 1.5.0`
- `scikit-learn 1.0`

- 


Before running this dockerbuild , please verify you got docker daemon running.
```console
 $sudo systemctl start docker
```


### Running Docker

```console
$ docker build -t zoomcamp.
```







