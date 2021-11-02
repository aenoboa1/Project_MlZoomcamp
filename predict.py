import pickle
import xgboost 

model_file = 'XGB_model_1.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)
    
customer = {'hotel': 'City Hotel',
 'arrival_date_month': 'January',
 'meal': 'BB',
 'market_segment': 'Groups',
 'distribution_channel': 'TA/TO',
 'customer_type': 'Transient',
 'lead_time': 86,
 'arrival_date_year': 2016,
 'arrival_date_week_number': 4,
 'arrival_date_day_of_month': 22,
 'stays_in_weekend_nights': 0,
 'stays_in_week_nights': 1,
 'adults': 2,
 'children': 0.0,
 'babies': 0,
 'is_repeated_guest': 0,
 'previous_cancellations': 1,
 'previous_bookings_not_canceled': 0,
 'booking_changes': 0,
 'agent': 29,
 'company': 0,
 'days_in_waiting_list': 35,
 'adr': 85.0,
 'required_car_parking_spaces': 0,
 'total_of_special_requests': 0}


X = dv.transform([customer])

y_pred = model.predict_proba(X)[0, 1]

print('input',customer)
print('cancel probability',y_pred)
