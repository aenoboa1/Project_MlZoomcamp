#!/usr/bin/env python
# coding: utf-8

import requests



host = 'hotel-serving-env.eba-sf8r37gc.us-east-1.elasticbeanstalk.com.'
url = f'http://{host}/predict'


#url = 'http://localhost:9696/predict'

customer_id = 'Customer_1'
customer = {
    'hotel': 'Resort Hotel',
    'arrival_date_month': 'September',
    'meal': 'BB',
    'market_segment': 'Online TA',
    'distribution_channel': 'TA/TO',
    'customer_type': 'Transient',
    'lead_time': 171,
    'arrival_date_year': 2015,
    'arrival_date_week_number': 37,
    'arrival_date_day_of_month': 12,
    'stays_in_weekend_nights': 2,
    'stays_in_week_nights': 2,
    'adults': 1,
    'children': 0.0,
    'babies': 0,
    'is_repeated_guest': 0,
    'previous_cancellations': 0,
    'previous_bookings_not_canceled': 0,
    'booking_changes': 2,
    'agent': 240,
    'company': 0,
    'days_in_waiting_list': 0,
    'adr': 59.13,
    'required_car_parking_spaces': 0,
    'total_of_special_requests': 1
}


response = requests.post(url, json=customer).json()
print(response)

if response['cancel'] == True:
    print('Customer  %s will likely cancel the booking' % customer_id)
else:
    print('%s will not likely cancel the booking ' % customer_id)