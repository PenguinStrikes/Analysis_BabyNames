import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
from sklearn import linear_model
import matplotlib.pyplot as plt
sns.set_style('whitegrid')

def levenshtein_hand(str1, str2):
    # Get the length of each string
    lenx = len(str1) + 1
    leny = len(str2) + 1

    # Create a matrix of zeros
    matrx = np.zeros((lenx, leny))

    # Index the first row and column
    matrx[:,0] = [x for x in range(lenx)]
    matrx[0] = [y for y in range(leny)]

    # Loop through each value in the matrix
    for x in range(1, lenx):
        for y in range(1, leny):
            # If the two string characters are the same
            if str1[x-1] == str2[y-1]:
                matrx[x,y] = matrx[x-1,y-1]
            else:
                matrx[x,y] = min(matrx[x-1, y]+1, matrx[x-1,y-1]+1, matrx[x,y-1]+1)

    operations = matrx[lenx-1,leny-1]
    percent    = round((1-(operations/(max(lenx, leny)))) * 100, 2)
    return percent

def rhythm_score(word, frame, found):
    ''' Use levenshtein formula to get word sound familiarity
    '''
    if found == True:
        data = []
        for index, row in frame.iterrows():
            score = levenshtein_hand(word, row['phonetics_space'])
            data.append(score)

        frame['similarity_score'] = data
    else:
        frame['similarity_score'] = '-'
    return frame
        

def alliteration(word, frame):
    ''' Occurence of the same letter or sound at the beginning of words
    '''
    single = word[0].upper()
    double = word[0:2].upper()
    
    allit1, allit2 = [], []
    
    for index, row in frame.iterrows():
        if row['single'] == single:
            allit1.append('Y')
        else:
            allit1.append('N')
        if row['double'] == double:
            allit2.append('Y')
        else:
            allit2.append('N')
    
    frame['single_allit'] = allit1
    frame['double_allit'] = allit2
    
    return frame


def last_sound(phone, frame, found):
    if found == True:
        data = []
        for index, row in frame.iterrows():
            if row['final_sound'] == phone:
                data.append('Y')
            else:
                data.append('N')
        frame['final_match'] = data
    else:
        frame['final_match'] = '-'
    return frame


def run_analysis(name, gender, remove, hyphen):
    
    last = pd.read_csv('../data/interim/lastnames.csv')
    frst = pd.read_csv('../data/interim/frstnames_pop.csv')
    
    name = name.upper()
    try:
        phon_name  = last[last['Surname'] == name]['phonetics_space'].values[0]
        final_phon = last[last['Surname'] == name]['final_sound'].values[0]
        print('Lastname in dictionary:', phon_name, final_phon)
        found = True
    except:
        print('Lastname not in dictionary:')
        phon_name  = 'NA'
        final_phon = 'NA'
        found = False
    
    df = alliteration(name, frst)
    df = rhythm_score(phon_name, df, found)
    df = last_sound(final_phon, df, found)
    
    if hyphen == False:
        df = df[df['Name'].str.contains('-') == False]
        
    df = df[df['Sex'] == gender]
    df['Surname'] = name
    
    if remove == 'Yes':
        df = df[(df.final_match == 'N') & (df.single_allit == 'N')]
        
    df = df[['Name','Surname','Rank','Predicted Rank','Count','Predicted Change','trend_profile',
             'similarity_score','phonetics_space','phonetics_source']]
    df = df.sort_values('Rank', ascending=True)
    return df


def load_transform_trend():
    named_data = pd.read_csv('../data/raw/names/firstnames.csv')
    named_data['Count'] = named_data['Count'].str.replace(',','')
    named_data['Count'] = named_data['Count'].astype(int)
    named_data = named_data[named_data['Count'] >= 3]
    named_data['Name'] = named_data['Name'].str.upper()
    named_data['Name'] = named_data['Name'].str.replace(' ','')
    named_data['Name'] = named_data['Name'].str.replace("'","")
    named_data['Name'] = named_data['Name'].str.replace(".","")
    
    totals = (
        named_data[['Name','Count','Sex','Year']]
        .groupby(['Name','Sex','Year'])
        .sum().reset_index()
    )

    totals['Rank'] = totals.groupby(['Year','Sex'])['Count'].rank(ascending=False, method='first')
    return totals


def year_data(df, Year1, Year2, sx):
    y1 = df[(df.Year == Year1) & (df.Sex == sx) & (df.Rank <= 10)].sort_values('Rank')
    y2 = df[(df.Year == Year2) & (df.Sex == sx) & (df.Rank <= 10)].sort_values('Rank')
    
    fig, ax = plt.subplots(figsize=(9,8))
    plt.scatter([0]*10, y1['Rank'], s=350, label=y1['Name'], zorder=10, c='#737373')
    plt.scatter([1]*10, y2['Rank'], s=350, label=y2['Name'], zorder=10, c='#737373')

    for name in y1['Name'].values:
        try:
            ins = list(y1['Name'].values).index(name) + 1
            ind = list(y2['Name'].values).index(name) + 1
            if ind < ins:
                c = '#42B6C4'
            elif ind > ins:
                c = '#FDB24C'
            else:
                c = '#737373'
            plt.plot([0,1],[ins,ind], zorder=1, c=c, linewidth=3)
        except:
            continue

    for i, txt in enumerate(y1['Name'].values):
        ax.annotate(txt, (-0.38,i+1.16), fontsize=16)

    for i, txt in enumerate(y2['Name'].values):
        ax.annotate(txt, (1.38,i+1.16), horizontalalignment='right', fontsize=16)

    plt.xlim(-0.4,1.4)
    plt.gca().invert_yaxis()
    sns.despine(left=True, bottom=True)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.savefig('../report/fig/'+str(Year1)+'_'+str(Year2)+'_'+sx+'.png', bbox_inches='tight', dpi=500)
    plt.show()

    return y1, y2

def produce_trends(totals):
    dte = list(range(2000,2018))
    X = np.array(dte).reshape(-1,1)
    lst = [i for i in range(0, len(dte), 4)]
    lst.append(18)
    lst = list(np.diff(lst)[::-1])

    master = []
    names_iter = totals[['Name','Sex']].drop_duplicates()

    for index, row in tqdm.tqdm(names_iter.iterrows()):
        df_name  = totals[(totals.Name == row['Name']) & (totals.Sex == row['Sex'])]
        df_name  = df_name[['Year','Count']]
        dte_dict = df_name.set_index('Year').T.to_dict('list')

        # Fill in the blanks with zero values for trending
        data_list = [0 if x not in dte_dict else dte_dict[x][0] for x in dte]

        # Batch linear regression on smaller blocks of data
        strt = 0
        min_pred = []
        min_scor = []
        min_coef = []

        for gap in lst:
            reg = linear_model.LinearRegression()
            reg.fit(X[strt:strt+gap], data_list[strt:strt+gap])
            for num in [int(i) for i in reg.predict(X[strt:strt+gap])]:
                min_pred.append(num)
            min_scor.append(float(str(round(reg.score(X[strt:strt+gap], data_list[strt:strt+gap]),2))))
            min_coef.append(float(str(round(reg.coef_[0],2))))
            strt = strt + gap
            prediction_fut = int(reg.predict(2018))
            
        profile = trend_profile(min_coef)

        # Fit a linear regression model to the data
        reg = linear_model.LinearRegression()
        reg.fit(X, data_list)
        

        # Append all the attributes to a list
        data = (row['Name'],
                row['Sex'],
                data_list,
                profile,
                [int(i) for i in reg.predict(X)],
                round(reg.score(X, data_list),2),
                round(reg.coef_[0],2),
                prediction_fut,
                min_pred, min_scor, min_coef)

        master.append(data)

    df_trends = pd.DataFrame(master, columns=['Name','Sex','trend_data','trend_profile',
                                              'trend_pred','trend_score','trend_coef',
                                              'trend_2018','batch_pred','batch_score','batch_coef'])
    return df_trends, dte


def trend_profile(coefs):
    trend = ['D' if i < -20 else 'G' if i > 20 else '-' for i in coefs]

    if trend[-1] == 'G':
        if trend.count('G') > 1:
            profile = 'GROWING BOOM'
        else:
            profile = 'RECENT BOOM'
    elif trend[-1] == 'D':
        if trend.count('D') > 1:
            profile = 'CONTINUED DECLINE'
        else:
            profile = 'RECENT DECLINE'
    else:
        if trend.count('D') > 2:
            profile = 'DECLINE STABILISING'
        elif trend.count('G') > 2:
            profile = 'GROWTH STABILISING'
        else:
            profile = 'CURRENTLY STABLE'

    return profile