from pathlib import Path
import json
import pandas as pd
import streamlit as st
import joblib

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='StaySure | Cancellation Risk',page_icon='🏨',layout='wide')
st.markdown('''<style>.block-container{padding-top:2rem}h1,h2,h3{color:#23345d}div[data-testid="stMetric"]{background:#f2f5ff;padding:18px;border-radius:14px}</style>''',unsafe_allow_html=True)
st.title('🏨 StaySure — Hotel Cancellation Intelligence')
st.caption('A machine-learning decision-support demo · Hotel Booking Demand dataset')
model_path=ROOT/'model.joblib'
if not model_path.exists():
    st.error('Model not trained yet. Run `python train.py` first, then redeploy with model.joblib included.')
    st.stop()
model=joblib.load(model_path)
results=json.loads((ROOT/'results.json').read_text()) if (ROOT/'results.json').exists() else {}
page=st.sidebar.radio('Explore the project',['Overview','Predict a booking','Model performance','About & limitations'])
if page=='Overview':
    st.header('Prevent avoidable revenue loss')
    st.write('Estimate cancellation risk before arrival and explore a suggested deposit policy. This is a decision-support prototype, not an automatic booking decision system.')
    a,b,c=st.columns(3)
    a.metric('Bookings in dataset',f"{results.get('dataset_rows',0):,}")
    b.metric('Observed cancellation rate',f"{100*results.get('cancel_rate',0):.1f}%")
    c.metric('Held-out ROC-AUC',f"{results.get('test',{}).get('roc_auc',0):.3f}")
    st.info('Use the sidebar to test a booking and review model performance.')
elif page=='Predict a booking':
    st.header('New booking · risk estimate')
    with st.form('booking'):
        a,b,c=st.columns(3)
        with a:
            hotel=st.selectbox('Hotel type',['City Hotel','Resort Hotel'])
            lead_time=st.number_input('Days before arrival',0,730,90)
            arrival_date_month=st.selectbox('Arrival month',['January','February','March','April','May','June','July','August','September','October','November','December'])
            stays_in_weekend_nights=st.number_input('Weekend nights',0,30,1)
            stays_in_week_nights=st.number_input('Week nights',0,60,2)
            adults=st.number_input('Adults',1,10,2)
            children=st.number_input('Children',0,10,0)
        with b:
            babies=st.number_input('Babies',0,10,0)
            meal=st.selectbox('Meal',['BB','HB','FB','SC','Undefined'])
            market_segment=st.selectbox('Market segment',['Online TA','Offline TA/TO','Direct','Groups','Corporate','Complementary','Aviation','Undefined'])
            distribution_channel=st.selectbox('Distribution channel',['TA/TO','Direct','Corporate','GDS','Undefined'])
            is_repeated_guest=int(st.checkbox('Repeat guest'))
            previous_cancellations=st.number_input('Previous cancellations',0,50,0)
            previous_bookings_not_canceled=st.number_input('Previous completed bookings',0,100,0)
            reserved_room_type=st.selectbox('Reserved room type',list('ABCDEFGHLP'))
        with c:
            booking_changes=st.number_input('Booking changes',0,30,0)
            deposit_type=st.selectbox('Current deposit type',['No Deposit','Non Refund','Refundable'])
            customer_type=st.selectbox('Customer type',['Transient','Transient-Party','Contract','Group'])
            adr=st.number_input('Average daily rate',0.0,10000.0,110.0)
            required_car_parking_spaces=st.number_input('Parking spaces',0,10,0)
            total_of_special_requests=st.number_input('Special requests',0,10,1)
        submit=st.form_submit_button('Estimate cancellation risk',type='primary')
    if submit:
        values=locals()
        fields=results.get('feature_columns',[])
        row=pd.DataFrame([{f:values[f] for f in fields}])
        prob=float(model.predict_proba(row)[0,1])
        level='Low' if prob<.3 else 'Moderate' if prob<.6 else 'High'
        policy={'Low':'No additional deposit suggested','Moderate':'Consider a partial deposit','High':'Consider a refundable or stricter deposit policy'}[level]
        st.metric('Estimated cancellation probability',f'{prob:.1%}')
        st.progress(prob)
        st.subheader(f'{level} risk · {policy}')
        st.caption('Policy thresholds are illustrative, not optimized for financial impact. Review fairness, customer experience and local rules before use.')
elif page=='Model performance':
    st.header('Model evaluation')
    st.write('Validation results below are from training-only splits; the held-out test set was evaluated once after model selection.')
    if results.get('validation_models'):
        st.dataframe(pd.DataFrame(results['validation_models']),use_container_width=True,hide_index=True)
    st.subheader('Final held-out test results')
    st.json(results.get('test',{}))
    st.write('Confusion matrix (rows: actual 0/1, columns: predicted 0/1)')
    st.write(results.get('confusion_matrix',[]))
    st.caption(f"Best CV ROC-AUC: {results.get('cv_best_auc','N/A')}")
else:
    st.header('Project scope and limitations')
    st.write('Predicts is_canceled using booking-time attributes. We intentionally exclude reservation_status, reservation_status_date, assigned_room_type and other post-outcome fields to reduce target leakage.')
    st.write('Observational data may be outdated or unrepresentative of a particular hotel. Probability calibration, cost-sensitive decisions, monitoring and fairness checks are required before real-world deployment.')
    st.markdown('**Source:** [Hotel Booking Demand — TidyTuesday mirror](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-02-11)')
