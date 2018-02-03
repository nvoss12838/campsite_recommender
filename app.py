from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import requests
from xml.etree import ElementTree
from sklearn.metrics.pairwise import linear_kernel
import geopy.distance
import sqlite3
from mapboxgl.viz import *
from mapboxgl.utils import *
from mapboxgl.colors import *
import plotly.offline as py_off
from plotly.graph_objs import *
import plotly
import mapbox
import geojson
from plotly.graph_objs import *
import config
from mapbox import Geocoder




app = Flask(__name__)
geocoder= Geocoder(access_token=config.api_key)

def find_similar_bin(properties,site_index):
    diff=[]
    for i,index in enumerate(properties):
        if i!=site_index:
            diff.append((i,np.count_nonzero(properties[site_index]!=index)))
    return sorted(diff, key = lambda x: int(x[1]))

def find_similar(tfidf_matrix, index):
    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
    return [(index, cosine_similarities[index]) for index in related_docs_indices]

def rank(site,df,df_des,home,distance):
    '''
    given a site index give a ranked list of similar sites
    inputs:
        sit,
        df: main table of sites
        df: associated description of sites
        number_of_sites: number of sites to return
    outputs:
        ranked similarity
    '''
    name = site[:]
    site = df[df.facilityName==site].index.values[0]
    #calculate the distance to from the selected site to all other sites
    response = geocoder.forward(home)
    site_coord = response.geojson()['features'][0]['center']
    df['Distance']=[geopy.distance.vincenty([site_coord[1],site_coord[0]], [i['latitude'],i['longitude']]).miles for sites,i in df.iterrows()]
    #subset by distance
    #description_rank:
    tf = TfidfVectorizer(ngram_range=(1,3), min_df = 0, stop_words = 'english')
    tfidf_desc =  tf.fit_transform(df_des.description.values)
    rank = find_similar(tfidf_desc,site)
    #map the index back to the original values
    df_desc_rank = pd.DataFrame({'Desc_rank':np.arange(len(rank)),'description':df_des.description},index=[a[0] for a in rank])
    #activity rank:
    tfidf_desc =  tf.fit_transform([' '.join(activities) for activities in df_det.activities])
    rank = find_similar(tfidf_desc,site)
    #map the index back to the original values
    df_act_rank = pd.DataFrame({'Act_rank':np.arange(len(rank))},index=[a[0] for a in rank])
    #ammenity_rank:
    df.replace(('Y', 'N'), (1, 0), inplace=True)
    properties=[]
    for index,sites in df.iterrows():
        properties.append([sites['sitesWithAmps'],sites['sitesWithPetsAllowed'],sites['sitesWithSewerHookup'],sites['sitesWithWaterHookup']])
    rank = find_similar_bin(properties,site)
    df_amen_rank = pd.DataFrame({'Amen_rank':np.arange(len(rank))},index=[a[0] for a in rank])
    df_rank = df.join([df_act_rank,df_desc_rank,df_amen_rank])
    df_rank['rank']=df_rank['Amen_rank']*0.1+df_rank['Act_rank']*0.45+df_rank['Desc_rank']*0.45
    df_rank = df_rank[df_rank['Distance']<distance]
    orig_row = df_des.loc[[site]].rename(lambda x: 'original')
    orig_row['facilityName'] = df.loc[site].facilityName
    print(orig_row)
    #df_des = df_des[df_des['facilityID'].isin(df['facilityID'])]
    #print orig_row
    return orig_row[['facilityName','description']],df_rank.sort(columns='rank')


conn = sqlite3.connect("/home/nvoss/campin.site/campsites.db")
df = pd.read_sql('SELECT * from campsites',con=conn)


import pickle
df_det = pd.read_pickle('/home/nvoss/campin.site/description.p')


@app.route('/', methods =['GET','POST'])
def index():
    return render_template('home.html')

@app.route('/campsites', methods=['GET','POST'])
def campsites():
    return render_template('index.html',df=df)

@app.route('/get_campsites')
def get_campsites():
    state = request.args.get('state')
    if state:
        sub_df = df[df['state'] == state]
        sub_df.sort_values(by='facilityName',inplace=True)
        id_name_color = [("","Select a Campsite...","white")] + list(zip(list(sub_df.index),list(sub_df['facilityName'])))
        data = [{"id": str(x[0]), "name": str(x[1])} for x in id_name_color]
        # print(data)
    return jsonify(data)

@app.route('/campsite_recommendations', methods=['GET','POST'])
def campsite_recommendations():
    campsite = request.form['campsite']
    home = request.form['home']
    distance = float(request.form['distance'])
    if campsite == '':
        return 'You must select a campsite.'
    if campsite != '':
        row,results_df = rank(campsite,df,df_det,home,distance)
        #print(results_df)
        #if len(results_df==0):
        #return 'No campsite in search radius'
        print(results_df['description'])
        #token = 'pk.eyJ1IjoibnZvc3MxMjgzOCIsImEiOiJjamJlM3NjNjkyZzRvMzJwZXBsY2tveTVmIn0.rR3InDFY2dvoNjqUAdrGgg'
        token = config.api_key
        ranked = [str(i+1) for i,v in enumerate(results_df.index)]
        text1 = [a + ' ' +  b   for a,b in zip(ranked,results_df['facilityName'])]
        graphs = [dict(
            data = Data([

                    (Scattermapbox(
                        mode='markers',
                        lat=results_df.latitude,
                        lon=results_df.longitude,
                        marker=Marker(color='black',size=20),
                            hoverinfo='text',text = text1)),
                    (Scattermapbox(
                                mode='markers+text',
                                lat=results_df.latitude,
                                lon=results_df.longitude,
                                marker=Marker(
                                    size=0),hoverinfo='none',text = ranked,
                                    textfont={"color":"white"})),
                    (Scattermapbox(
                        mode='markers',
                        lat=results_df.latitude,
                        lon=results_df.longitude,
                        marker=Marker(color='black',size=0),
                            hoverinfo='text',text = text1))
                    ]),
            layout = Layout(
                margin=dict(t=0,b=0,r=0,l=0),
                autosize=True,
                hovermode='closest',
                showlegend=False,
                mapbox=dict(
                    accesstoken=token,
                    bearing=0,
                    center=dict(
                        lat=np.mean(pd.to_numeric(results_df.latitude)),
                        lon=np.mean(pd.to_numeric(results_df.longitude))
                    ),
                    pitch=0,
                    zoom=7,
                    style='outdoors'
                ),
            ))]
        ids = ['graph-{}'.format(i) for i, _ in enumerate(graphs)]
        graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)


        return render_template('campsite_recommendations.html',row=row,ids=ids,results_df=results_df[['facilityName','description']],graphJSON=graphJSON)
    return 'You must select a campsite!.'






if  __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)
