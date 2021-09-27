import matplotlib.pyplot as plt
from pandas.core.indexes.datetimes import date_range
import plotly.express as px
import plotly.graph_objects as go
import statsmodels as sm
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

###
# Read Data
###
transactions = pd.read_csv('transactions.csv', parse_dates=['transactions'])
support = pd.read_csv('support.csv', parse_dates=['created_date', 'solved_date'])


###
# Manipulate Data
###
support['time_resolved'] = support['solved_date'] - support['created_date']
support['days'] = support['time_resolved'].dt.days
support['month'] = support['created_date'].dt.strftime("%Y-%m-01")
# support['created_date'].dt.resample('BM')
support['year'] = support['solved_date'].dt.strftime('%Y')
support['created_weekday'] = support['created_date'].dt.strftime('%a')
support['solved_weekday'] = support['solved_date'].dt.strftime('%a')
support['is_weekend'] = np.where(support['created_weekday'].isin(['Fri', 'Sat']), 'Weekend', 'Weekday')
## Must be a chat bot
support['is_bot'] = np.where(support["assignee_id"].isin([655155207, 655155208]), "Bot", "Person")
support['is_solved'] = np.where(support['year']!='1902', 1, 0)
support['is_assigned'] = np.where(support['assignee_id'].isin([0, 1]), 0, 1)

assigned_df = support[support['is_assigned'] == 1]
a_solved_df = assigned_df[assigned_df['is_solved']==1]


###
# Explore Support Data
###

## Look at date ranges
a_solved_df.created_date.describe()
a_solved_df.solved_date.describe()

# '1902' must mean it wasn't solved; 1073 weren't solved in this year.
assigned_df.groupby('is_solved')['id'].count()

# Number of support tickets per day is going up.
plt_df = a_solved_df.groupby('created_date')['id'].count().reset_index()

# Around 1600 support tickets a day; somewhat bimodal.
plt.hist(x = 'id', bins = 40, data = plt_df)
plt.show()

# Number of support tickets per day is increasing in the 365 day period.
plt.plot_date('created_date', 'id', data = plt_df, fmt='o', tz=None, xdate=True, ydate=False)
plt.gcf().autofmt_xdate
plt.show()

## Number of tickets per day: Person vs. Bot
plt_df = assigned_df.groupby(['created_date', 'is_bot'])['id'].count().reset_index()
fig=px.scatter(
    plt_df,
    x='created_date', 
    y='id', 
    trendline='ols', 
    color='is_bot', 
    title='Number of Daily Bookings vs. Support Tickets Created',
    labels={
        'bookings':'Number of Daily Bookings',
        'noBot_cases':'Number of Support Cases Created'
    })
fig.update_layout(
    plot_bgcolor='white', 
    legend_bordercolor='grey',
    legend_orientation='h',
    legend_y=1.1,
    legend_yanchor='top',
    legend_title_text='')
fig.update_yaxes(rangemode='tozero', tickformat = ".2s")
fig.update_xaxes(rangemode='tozero')
fig.show()

# Number of support people per month.
plt_df = assigned_df.groupby('month')['assignee_id'].nunique().reset_index()

# Average number of support people assigned to tickets per month = 2,257
plt_df.assignee_id.mean()

plt.plot_date('month', 'assignee_id', data = plt_df, fmt='o', tz=None, xdate=True, ydate=False)
plt.show()

# Number of people per day
plt_df = assigned_df.groupby('created_date')['assignee_id'].nunique().reset_index()

# Average number of support people assigned to tickets per day = 775
plt_df.assignee_id.describe()

plt.plot_date('created_date', 'assignee_id', data = plt_df, fmt='o', tz=None, xdate=True, ydate=False)
plt.show()

# Number of Tickets per Person per Day
plt_df = a_solved_df.groupby(['assignee_id', 'is_bot', 'month', 'created_date'])['id'].count().reset_index()

fig=px.scatter(
    plt_df,
    x='created_date', 
    y='id',
    color='is_bot', 
    color_discrete_sequence = ['rgb(55,126,184)','rgb(153,153,153)'],
    title='Number of Support Cases Assigned',
    labels={
        'created_date':'Date',
        'id':'Number of Support Cases Assigned'
    })
fig.update_layout(
    plot_bgcolor='white', 
    legend_bordercolor='grey',
    legend_orientation='h',
    legend_y=1,
    legend_yanchor='top',
    legend_title_text='')
fig.update_yaxes(rangemode='tozero', tickformat = ".2s")
fig.update_xaxes(rangemode='tozero')
fig.show()

# Number of days to solve
plt_df = a_solved_df.groupby(['created_date', 'is_bot'])['days'].mean().reset_index()
px.scatter(plt_df, x='created_date', y='days', color='is_bot').show()

# Number of assigned grows as we approach March 2020
plt_df = assigned_df.groupby('created_date').agg({'id':[np.mean, np.max]}).reset_index()

df = plt_df.xs('amax', axis=1, level=1)
plt.hist(df, bins = 100)
plt.show()

numTickets_df = assigned_df.groupby('id')['assignee_id'].count().reset_index()
numTickets_df[numTickets_df.id > 25]

plt.bar(x = 'id', height = 'assignee_id', data = numTickets_df[numTickets_df.id > 25])
plt.show()


# Average time to solve
# Number of support tickets per day is going up.
resolved_df = support[support.is_solved == True]
resolved_df2 = resolved_df[resolved_df.days < 10]
resolved_df2['is_France'] = np.where(resolved_df2['requester_language']=='french', True, False)

plt_df = resolved_df2.groupby(['created_date', 'is_France'])['days'].mean().reset_index()

## Number of support tickets per day is increasing in the 365 day period.
fig = px.scatter(plt_df, x="created_date", y='days', color='is_France')
fig.show()


## MAX number of tickets per person per day
resolved_df['is_France'] = np.where(resolved_df['requester_language']=='french', True, False)

plt_df = resolved_df.groupby(['created_date', 'is_France'])['days'].max().reset_index()

fig = px.scatter(plt_df, x="created_date", y='days', color='is_France')
fig.show()


####
# Explore Transactions
####
transactions['is_forecast']=np.where(transactions['transactions']<'2020-03-01', 'Actual', 'Forecast')
colorsIdx={0:'rgb(215,48,39)', 1:'rgb(128,128,128)'}
cols=transactions['is_forecast'].map(colorsIdx)

fig=px.scatter(
    transactions,
    x='transactions', 
    y='bookings',
    color='is_forecast', 
    color_discrete_sequence = ['rgb(55,126,184)','rgb(153,153,153)'],
    title='Number of Bookings: 2019-2020',
    labels={
        'transactions':'Date',
        'bookings':'Number of Bookings'
    })
fig.update_layout(
    plot_bgcolor='white', 
    legend_bordercolor='grey',
    legend_orientation='h',
    legend_y=1,
    legend_yanchor='top',
    legend_title_text='')
fig.update_yaxes(rangemode='tozero', tickformat = ".2s")
fig.update_xaxes(rangemode='tozero')
fig.show()



fig=go.Figure()
fig.add_trace(
    go.Scatter(
        mode='markers',
        marker_symbol='circle',
        x=transactions.transactions,
        y=transactions.checkins,
        marker=dict(
            size=5,
            color=cols)
    )
)
fig.add_trace(
    go.Scatter(
        mode='markers',
        marker_symbol='circle-open',
        x=transactions.transactions,
        y=transactions.bookings,
        marker=dict(
            color=cols,
            size=5
        )
    )
)
fig.update_layout(
    plot_bgcolor='white', 
    legend_bordercolor='grey',
    legend_orientation='h',
    legend_y=1,
    legend_yanchor='top',
    legend_title_text='')
fig.update_yaxes(rangemode='tozero', tickformat = ".2s")
fig.update_xaxes(rangemode='tozero')

fig.show()

## Bookings vs. Checkins
fig=go.Figure()
fig.add_trace(
    go.Scatter(
        mode='markers',
        marker_symbol='circle',
        x=transactions.bookings,
        y=transactions.checkins,
        marker=dict(
            size=5,
            color=cols)
    )
)
fig.show()


###
# Explore Relationships between Bookings and Support Cases/People
###

# Create date table
dates=pd.date_range(start='2019-04-01', end=transactions.transactions.max(), freq='D')
dates_df=pd.DataFrame(dates, columns=['Dates'])
dates_df=pd.merge(left=dates_df, right=transactions, how='left', left_on='Dates', right_on='transactions')

# Aggregate support tickets and people by day
supportByDay=assigned_df.groupby(['created_date', 'month', 'created_weekday', 'is_weekend']).agg({'id':'count', 'assignee_id':'nunique'}).reset_index()
supportByDay=supportByDay.rename(columns={"id":"total_cases", "assignee_id":"total_people"})
dates_df=pd.merge(left=dates_df, right=supportByDay, how='left', left_on='Dates', right_on='created_date')

# Extract support data for non-bots (as identified) then aggregate
noBot_df=assigned_df[assigned_df.is_bot=='Person']
noBotByDay_df=noBot_df.groupby(['created_date', 'month', 'created_weekday', 'is_weekend']).agg({'id':'count', 'assignee_id':'nunique'}).reset_index()
noBotByDay_df=noBotByDay_df.rename(columns={"id":"noBot_cases", "assignee_id":"noBot_people"})

dates_df=pd.merge(left=dates_df, right=noBotByDay_df, how='left', left_on='Dates', right_on='created_date')

# FRENCH SUPPORT
france_support=assigned_df[assigned_df['requester_language']=='french']
france_noBot=france_support[france_support.is_bot=='Person']
france_noBotByDay=france_noBot.groupby(['created_date', 'month', 'created_weekday', 'is_weekend']).agg({'id':'count', 'assignee_id':'nunique'}).reset_index()
france_noBotByDay=france_noBotByDay.rename(columns={"id":"france_cases", "assignee_id":"france_people"})
dates_df=pd.merge(left=dates_df, right=france_noBotByDay, how='left', left_on='Dates', right_on='created_date')

# Filter for actual data
actuals_df = dates_df[dates_df['is_forecast']==0]
actuals_df['tickets_per']=actuals_df['noBot_cases']/actuals_df['noBot_people']

# Bookings vs. Cases
fig=px.scatter(
    actuals_df,
    x='bookings', 
    y='noBot_cases', 
    trendline='ols', 
    color='is_weekend', 
    color_discrete_sequence = ['rgb(55,126,184)','rgb(153,153,153)'],
    title='Number of Daily Bookings vs. Support Tickets Created',
    labels={
        'bookings':'Number of Daily Bookings',
        'noBot_cases':'Number of Support Cases Created'
    })
fig.update_layout(
    plot_bgcolor='white', 
    legend_bordercolor='grey',
    legend_orientation='h',
    legend_y=1.1,
    legend_yanchor='top',
    legend_title_text='')
fig.update_yaxes(rangemode='tozero', tickformat = ".2s")
fig.update_xaxes(rangemode='tozero')
fig.show()

# Bookings vs. People
fig=px.scatter(
    actuals_df,
    x='bookings',
    y='noBot_people',
    trendline='ols',
    labels={
        "bookings": "Number of Daily Bookings",
        "assignee_id_y": "Number of Support People"
    },
    title='Number of Daily Bookings vs. Number of Support People')
fig.update_layout(
    font_color="grey"
)  
fig.show()

## Bookings vs. France Cases
fig=px.scatter(
    actuals_df,
    x='bookings', 
    y='france_cases', 
    trendline='ols', 
    color='is_weekend', 
    color_discrete_sequence = ['rgb(55,126,184)','rgb(153,153,153)'],
    title='Number of Daily Bookings vs. Support Tickets Created by French Speaking Customers',
    labels={
        'bookings':'Number of Daily Bookings',
        'france_cases':'Number of Support Cases Created'
    })
fig.update_layout(
    plot_bgcolor='white', 
    legend_bordercolor='grey',
    legend_orientation='h',
    legend_y=1.1,
    legend_yanchor='top',
    legend_title_text='')
fig.update_yaxes(rangemode='tozero', tickformat = ".2s")
fig.update_xaxes(rangemode='tozero')
fig.show()

## France Cases vs. France support people
fig=px.scatter(
    actuals_df,
    x='france_cases', 
    y='france_people', 
    trendline='ols', 
    color_discrete_sequence = ['rgb(55,126,184)','rgb(153,153,153)'],
    title='Number of Daily Cases vs. Support People for French Speaking Customers',
    labels={
        'france_cases':'Number of Support Cases Created',
        'france_people':'Number of Support People'
    })
fig.update_layout(
    plot_bgcolor='white', 
    legend_bordercolor='grey',
    legend_orientation='h',
    legend_y=1.1,
    legend_yanchor='top',
    legend_title_text='')
fig.update_yaxes(rangemode='tozero', tickformat = ".2s")
fig.update_xaxes(rangemode='tozero')
fig.show()

# Bookings vs. Tickets per Person
px.scatter(actuals_df, x='bookings', y='tickets_per', trendline='ols', color='is_weekend_y', title='Number of Daily Bookings vs. Support Tickets Created').show()


###
# Linear Regression
###

# LINEAR REGRESSION WITH WEEKEND INDICATOR
y =actuals_df['noBot_people']
x = pd.DataFrame(actuals_df[['bookings', 'is_weekend_y']])
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 100) 

# Fit the model
reg=LinearRegression(fit_intercept=True).fit(x_train, y_train)

# Predict Y from Test set
y_pred = reg.predict(x_test)

# Print coefficients and model metrics
coefficients = pd.concat([pd.DataFrame(x_train.columns),pd.DataFrame(np.transpose(reg.coef_))], axis = 1)
print(coefficients, reg.intercept_)
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

# Graph y and y_predictions vs. x
colorsIdx={0:'rgb(215,48,39)', 1:'rgb(215,148,39)'}
cols=x_test['is_weekend_y'].map(colorsIdx)
fig=go.Figure()
fig.add_trace(
    go.Scatter(
        mode='markers',
        marker_symbol='square',
        x=x_test['bookings'],
        y=y_test,
        marker=dict(
            size=8,
            color=cols)
    )
)
fig.add_trace(
    go.Scatter(
        mode='markers',
        marker_symbol='circle-open',
        x=x_test['bookings'],
        y=y_pred,
        marker=dict(
            color=cols,
            size=8
        )
    )
)
fig.show()

# LINEAR REGRESSION WITHOUT WEEKEND INDICATOR
y =actuals_df['noBot_people']
x = pd.DataFrame(actuals_df[['bookings']])
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 100) 

# Fit model
reg=LinearRegression(fit_intercept=True).fit(x_train, y_train)

# Predict Y
y_pred = reg.predict(x_test)

# Print Coefficients
coefficients = pd.concat([pd.DataFrame(x_train.columns),pd.DataFrame(np.transpose(reg.coef_))], axis = 1)
reg.intercept_
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

# Graph y and y predictions vs. X
fig=go.Figure()
fig.add_trace(
    go.Scatter(
        mode='markers',
        marker_symbol='square',
        x=x_test['bookings'],
        y=y_test
    )
)
fig.add_trace(
    go.Scatter(
        mode='markers',
        marker_symbol='circle-open',
        x=x_test['bookings'],
        y=y_pred
    )
)
fig.show()


###
#  FRANCE - Linear Regression
###

# WITH WEEKEND INDICATOR
y =actuals_df['france_people']
x = pd.DataFrame(actuals_df[['bookings', 'is_weekend_y']])
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 100) 

# Fit Model
reg=LinearRegression(fit_intercept=True).fit(x_train, y_train)

# Predict Y
y_pred = reg.predict(x_test)

# Print Coefficients
coefficients = pd.concat([pd.DataFrame(x_train.columns),pd.DataFrame(np.transpose(reg.coef_))], axis = 1)
print(coefficients, reg.intercept_)
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

# Graph y, predictions vs. x
colorsIdx={0:'rgb(215,48,39)', 1:'rgb(215,148,39)'}
cols=x_test['is_weekend_y'].map(colorsIdx)
fig=go.Figure()
fig.add_trace(
    go.Scatter(
        mode='markers',
        marker_symbol='square',
        x=x_test['bookings'],
        y=y_test,
        marker=dict(
            size=8,
            color=cols)
    )
)
fig.add_trace(
    go.Scatter(
        mode='markers',
        marker_symbol='circle-open',
        x=x_test['bookings'],
        y=y_pred,
        marker=dict(
            color=cols,
            size=8
        )
    )
)
fig.show()



###
#  FUTURE FRANCE SUPPORT PEOPLE
###

# Forecasted bookings and associated data
forecast_test = dates_df[dates_df['is_forecast']==1]
forecast_test['weekday'] = forecast_test['Dates'].dt.strftime('%a')
forecast_test['is_weekend'] = np.where(forecast_test['weekday'].isin(['Fri', 'Sat']), 1, 0)

fc_test = forecast_test[['bookings', 'is_weekend']]

# Predict support people needed from forecasted bookings
france_pred = reg.predict(fc_test)

forecast_test['france_people']=france_pred

# Union actuals and forecast back together
plt_df = pd.concat([actuals_df, forecast_test])
plt_df['weekday'] = plt_df['Dates'].dt.strftime('%a')
plt_df['is_weekend'] = np.where(plt_df['weekday'].isin(['Fri', 'Sat']), 1, 0)

# Plot actuals and forecast with bookings and support people
colorsIdx={0:'rgb(215,48,39)', 1:'rgb(215,148,39)'}
cols=plt_df['is_weekend'].map(colorsIdx)
fig=go.Figure()
fig.add_trace(
    go.Scatter(
        mode='markers',
        marker_symbol='square',
        x=plt_df['Dates'],
        y=plt_df['france_people'],
        marker=dict(
            size=8,
            color=cols)
    )
)
fig.show()
