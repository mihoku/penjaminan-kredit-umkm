import flask
from datetime import datetime, timedelta
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle
import pathlib
from controls import monthCode, econSector, sectorColor, sectorTxtColor

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("dataset").resolve()
MODEL_PATH = PATH.joinpath("model").resolve()

#styling/css
external_stylesheets = ['https://dash-gallery.plotly.host/dash-oil-and-gas/assets/s1.css','https://dash-gallery.plotly.host/dash-oil-and-gas/assets/styles.css']

server = flask.Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server=server)

#read dataset
df = pd.read_csv(DATA_PATH.joinpath('dataset-predictive-NPL-UMKM.csv'),low_memory=False)

#import model 
model_rf = pickle.load(open(MODEL_PATH.joinpath("penjaminan_predictive_UMKM_2.sav"), "rb"))

pre_df = df[df.Tahun == 2020]
fil_df = pre_df[pre_df.Bulan=="Jun"]
row_take = fil_df[fil_df.SektorEkonomi==econSector[0]]

#populate average 2019 data
prev_df = df[df.Tahun != 2020]
summavg_df = prev_df.groupby('SektorEkonomi', as_index=False).agg({"percentNPL":"mean"})

avg_2019 = []  
    
for i in np.arange(18):
    taken_df = summavg_df[summavg_df.SektorEkonomi==econSector[i]]
    avg_2019.append(taken_df['percentNPL'].values[0]*100)

#form generation function
def generate_form(i):
    prefiltered_df = df[df.Tahun == 2020]
    filtered_df = prefiltered_df[prefiltered_df.Bulan=="Jun"]
    data = filtered_df[filtered_df.SektorEkonomi==econSector[i]]
    return html.Div([
        html.P(econSector[i], style={'color': sectorTxtColor[i], 'font-weight':'bold'}),
        html.P("Penyaluran Kredit", style={'color': sectorTxtColor[i]}),
        dcc.Input(
            id="sector_form_{}".format(str(i)),
            type="number",
            value=data['valueChannel'].values[0]*1000000000,
            debounce=True
        ),
        html.P("Proyeksi NPL", style={'color': sectorTxtColor[i]}),
        html.H1(id ="sector_NPL_{}".format(str(i)) , style={'color': sectorTxtColor[i], 'font-weight':'bold', 'font-size':'44px'}),
        html.P(id="sector_NPL_val_{}".format(i) , style={'color': sectorTxtColor[i]})
        ],className="three columns pretty_container", style={'width': '98%', 'background-color':sectorColor[i]})

#form generation function untuk evaluasi IJP
def generate_form_eval_IJP(i):
    prefiltered_df = df[df.Tahun == 2020]
    filtered_df = prefiltered_df[prefiltered_df.Bulan=="Jun"]
    data = filtered_df[filtered_df.SektorEkonomi==econSector[i]]
    return html.Div([
        html.P(econSector[i], style={'color': sectorTxtColor[i], 'font-weight':'bold'}),
        html.P("Penyaluran Kredit", style={'color': sectorTxtColor[i]}),
        dcc.Input(
            id="sector_form2_{}".format(str(i)),
            type="number",
            value=data['valueChannel'].values[0]*1000000000,
            debounce=True
        ),
        html.P("Proyeksi NPL", style={'color': sectorTxtColor[i]}),
        html.H1(id ="sector_NPL2_{}".format(str(i)) , style={'color': sectorTxtColor[i], 'font-weight':'bold', 'font-size':'44px'}),
        html.P(id="sector_NPL_val2_{}".format(i) , style={'color': sectorTxtColor[i]})
        ],className="three columns pretty_container", style={'width': '98%', 'background-color':sectorColor[i]})

#form generation function untuk evaluasi sektor terdampak
def generate_form_eval_sector(i):
    prefiltered_df = df[df.Tahun == 2020]
    filtered_df = prefiltered_df[prefiltered_df.Bulan=="Jun"]
    data = filtered_df[filtered_df.SektorEkonomi==econSector[i]]
    return html.Div([
        html.P(econSector[i], style={'color': sectorTxtColor[i], 'font-weight':'bold'}),
        html.P("Penyaluran Kredit", style={'color': sectorTxtColor[i]}),
        dcc.Input(
            id="sector_form3_{}".format(str(i)),
            type="number",
            value=data['valueChannel'].values[0]*1000000000,
            debounce=True
        ),
        html.P("Proyeksi NPL", style={'color': sectorTxtColor[i]}),
        html.H1(id ="sector_NPL3_{}".format(str(i)) , style={'color': sectorTxtColor[i], 'font-weight':'bold', 'font-size':'44px'}),
        html.P("Rata-rata Persentase NPL 2019" , style={'color': sectorTxtColor[i]}),
        html.H6(id="sector_compare_NPL3_{}".format(i) , style={'color': sectorTxtColor[i], 'font-size':'32px'}), 
        html.P(id="sector_NPL_val3_{}".format(i) , style={'color': sectorTxtColor[i]})
        ],className="three columns pretty_container", style={'width': '98%', 'background-color':sectorColor[i]})

app.layout = html.Div(children=[
    html.Div( #header div
            [
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("kemenkeu-logo.png"),
                            id="logo-image",
                            style={
                                "height": "100px",
                                "width": "auto",
                            },
                        ),#end of logo img tag
                        html.Div(
                            [
                                html.H3(
                                    "Penjaminan Kredit UMKM dalam rangka PEN",
                                    style={"margin-bottom": "0px", "font-weight":"bold"},
                                ),
                                html.H5(
                                    "Predictive Analytics Dashboard", style={"margin-top": "0px"}
                                ),
                            ]
                        )
                    ],
                    className="twelve columns",
                    id="title",
                )
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
            ),#end of header div
    html.Div( #main div
        [
            dcc.Tabs([
                dcc.Tab(label='Informasi Umum', children=[
                    
                        html.Div([ #latar belakang dan tujuan analisis
                            html.Div([
                                html.H5("Latar Belakang", style={"font-weight":"bold"}),
                                html.P("Dalam rangka mendukung kebijakan keuangan negara untuk penanganan pandemi Covid-19 dan pemulihan ekonomi nasional, Pemerintah melalui Peraturan Pemerintah nomor 43 tahun 2020 telah mengatur 4 (empat) modalitas untuk program pemulihan ekonomi nasional (PEN) yang meliputi penyertaan modal negara, penempatan dana, investasi pemerintah, dan penjaminan."),
                                html.P("Pada kegiatan penjaminan kredit modal kerja UMKM, pemerintah menugaskan BUMN dalam hal ini PT Jamkrindo dan PT Askrindo untuk bertindak sebagai penjamin bagi kredit modal kerja Usaha Mikro Kecil Menengah (UMKM). Program penjaminan ini sendiri bertujuan untuk meningkatkan minat perbankan dalam menyalurkan kredit kepada pelaku usaha agar mendapat kemudahan penjaminan saat mengajukan kredit. Selain itu, pemberian modal kerja pada UMKM penting dilakukan dalam membuat kegiatan usaha kembali menggeliat setelah terpuruk akibat dampak pandemi Covid-19."),
                                html.P("Pemerintah telah melakukan berbagai dukungan agar program penjaminan berjalan dengan baik. Pada tahun 2020, pemerintah telah menganggarkan sejumlah Rp6 T untuk memberikan dukungan pada program penjaminan pelaku usaha UMKM dengan rincian Rp5 T sebagai Subsidi Belanja IJP dan Rp1 T untuk dukungan penjaminan loss limit."),
                                html.P("Salah satu dukungan yang dilakukan pemerintah adalah membayarkan seluruh Imbal Jasa Penjaminan (IJP) yang seharusnya ditanggung oleh pelaku usaha sebagai kreditur. IJP yang dianggarkan pemerintah akan dibayarkan ke pihak penjamin sesuai dengan perhitungan yang telah ditetapkan. Salah satu faktor penentuan besaran IJP adalah adanya proyeksi non performing loan (NPL). Penentuan besaran rasio NPL yang akurat akan berpengaruh pada ketepatan jumlah penganggaran yang dilakukan pemerintah dalam alokasi pembayaran IJP. Pada penjaminan pemerintah pada pelaku usaha UMKM, penentuan tarif IJP didasari pada hasil metode perhitungan dan analisa PT Reindonesia Indonesia Utama (RIU) dengan mempertimbangkan proyeksi NPL.")
                                ],
                                className="pretty_container eight columns"),   
                            html.Div([
                                html.H5("Tujuan", style={"font-weight":"bold"}),
                                html.Div([
                                    html.P("Melihat sektor usaha UMKM yang paling terdampak dengan adanya pandemi COVID-19",style={"color":"#fff"})
                                    ],className = "pretty_container",
                                    style={"background-color":"#007bff"}),
                                html.Div([
                                    html.P("Memberikan usulan tarif IJP yang akan diberikan kepada Jamkrindo dan Askrindo sebagai lembaga penjamin program PEN",style={"color":"#fff"})
                                    ],className = "pretty_container",
                                    style={"background-color":"#28a745"}),
                                html.Div([
                                    html.P("Memberikan usulan anggaran belanja subsidi IJP dan Loss Limit yang sesuai dan tepat",style={"color":"#000"})
                                    ],className = "pretty_container",
                                    style={"background-color":"#ffc107"}),
                                html.P("Selain tujuan yang disebutkan di atas, analisis ini juga dapat bermanfaat untuk pelaksanaan kegiatan pengawasan yang dilakukan oleh Inspektorat Jenderal atas penjaminan program PEN. Hasil analisis dapat digunakan untuk melihat apakah tarif yang diusulkan oleh PT Reasuransi Indonesia Utama (PT RIU) telah disusun menggunakan prediksi NPL yang tepat dan anggaran yang diusulkan Direktorat Jenderal Pengelolaan Pembiayaan dan Risiko (DJPPR) sudah tepat.")
                                ],
                                id="predictiveDescription",
                                className="pretty_container four columns")
                            ],
                            className="row flex-display"), #end of latar belakang dan tujuan analisis
                    
                    html.Div([ #start of model chart
                        html.H5("Model Prediktif",style={"font-weight":"bold"}),
                        html.Div([
                            html.P("Model prediktif ini dikembangkan atas target utama yakni NPL penyaluran kredit kepada UMKM. Terdapat beberapa aspek yang menjadi prediktor dan secara umum terbagi ke dalam dua kelompok besar, yakni kondisi makroekonomi dan sektor ekonomi UMKM."),
                            html.P("Algoritma Random Forest Regression digunakan dalam pengembangan model prediktif tersebut. Random Forest merupakan jenis algoritma ensemble yang mengkombinasikan beberapa decision tree untuk membuat prediksi finalnya."),
                            html.P("Sumber data yang digunakan dalam pengembangan model prediktif ini adalah Laporan Statistik Perbankan Indonesia dari Otoritas Jasa Keuangan, serta Badan Pusat Statistik untuk indikator makroekonomi.")
                            ],style={'width': '40%','display':'inline-block'}),
                        html.Div([
                           html.Img(
                               src=app.get_asset_url("model-chart.png"),
                               id="scheme-image",
                               style={
                                   "height": "auto",
                                   "width": "80%",
                                   },
                               ),#end of logo img tag   
                            ],style={'text-align':'center','width': '60%','display':'inline-block'}),
                        ], className="pretty_container",style={'background-color':'#fff'}),
                    
                    html.Div([#start of aggregate credit channel and NPL graph div
                        html.H5("Total Penyaluran dan NPL Kredit UMKM Tahun 2011-2020 (dalam Rp Miliar)",style={"font-weight":"bold"}),
                        html.Div([
                            dcc.Graph(id='aggregate-channel-graph-with-slider',
                                      style={'height':500})   
                            ],style={'width': '50%','display':'inline-block'}),

                        html.Div([
                            dcc.Graph(id='aggregate-npl-graph-with-slider',
                                      style={'height':500})   
                            ],style={'width': '50%','display':'inline-block'}),

                        dcc.Slider(
                                id='aggregate-year-slider',
                                min=df['Tahun'].min(),
                                max=df['Tahun'].max(),
                                value=2020,
                                marks={str(year): str(year) for year in df['Tahun'].unique()},
                                step=None
                                ),
                        html.Br(),
                        html.Br()
                        ],className="pretty_container"),
                    
                    html.Div([ #start of sectoral credit channel and NPL graph div
                        html.Div([ #column div
                            html.H5("Penyaluran dan NPL Kredit UMKM per Sektor Ekonomi Tahun 2011-2020 (dalam Rp Miliar)",style={"font-weight":"bold"}),
                            html.Div([
                                dcc.Dropdown(
                                    id='econ-sector-selector',
                                    options=[{'label': i, 'value': i} for i in econSector],
                                    value='Perdagangan Besar dan Eceran'
                                    )                                
                                ],style={'width': '48%'}),
                            html.Div([
                                dcc.Graph(id='channel-graph-with-slider',
                                      style={'height':500})                                
                                ],style={'width': '60%','display':'inline-block'}),
                            html.Div([
                                dcc.Graph(id='channel-graph-with-slider-2',
                                      style={'height':500})                                
                                ],style={'width': '40%','display':'inline-block'}),
                            dcc.Slider(
                                id='year-slider',
                                min=df['Tahun'].min(),
                                max=df['Tahun'].max(),
                                value=2020,
                                marks={str(year): str(year) for year in df['Tahun'].unique()},
                                step=None
                                ),
                            html.Br()],
                            className="twelve columns")                
                        ],
                        id="display-graph-credit",
                        className="pretty_container row flex-display",
                        style={"margin-bottom": "25px", "background-color":"#fff"}
                        ),#end of credit channel and NPL graph div
                    html.Div([ #start of comparison on sectoral credit channel and NPL graph div
                        html.Div([ #column div
                            html.H5("Perbandingan Penyaluran dan NPL Kredit UMKM Antar Sektor Ekonomi Tahun 2011-2020", style={"font-weight":"bold"}),
                            html.Div([
                                dcc.Graph(id='channel-comparison-graph-with-slider',
                                      style={'height':600}),
                                dcc.Slider(
                                    id='year-slider-2',
                                    min=df['Tahun'].min(),
                                    max=df['Tahun'].max(),
                                    value=2020,
                                    marks={str(year): str(year) for year in df['Tahun'].unique()},
                                    step=None
                                    ),
                                html.Br(),
                                html.Br()                                
                                ],style={'width': '98%'}),
                            html.Div([
                                dcc.Graph(id='channel-comparison-graph-with-slider-2',
                                      style={'height':600}),
                                dcc.Slider(
                                    id='year-slider-3',
                                    min=df['Tahun'].min(),
                                    max=df['Tahun'].max(),
                                    value=2020,
                                    marks={str(year): str(year) for year in df['Tahun'].unique()},
                                    step=None
                                    ),
                                html.Br()                                
                                ],style={'width': '98%'}),
                            ],
                            className="twelve columns")                
                        ],
                        id="compare-graph-credit",
                        className="pretty_container row flex-display",
                        style={"margin-bottom": "25px"}
                        ),#end of comparison on sectoral credit channel and NPL graph div
                    ]),#end of first tab
                #######################tab limit#####################
                dcc.Tab(label='Evaluasi Sektor Terdampak', children=[

                    html.Div([ #start of total channeling and NPL row div
                        html.Div([ #start of column div for total SME Credit channeling
                            html.H5("Perbandingan Proyeksi NPL Kredit UMKM Antar Sektor Ekonomi", style={"font-weight":"bold", "color":"#000"}),
                            dcc.RadioItems(
                                id='npl_value_type',
                                options=[{'label': i, 'value': i} for i in ['Percentage', 'Value']],
                                value='Percentage',
                                labelStyle={'display': 'inline-block'}
                                ),
                            dcc.Graph(id='channel-comparison-graph-sector-affected',
                                      style={'height':600})
                            ], className="pretty_container twelve columns",
                            style={'text-align':'center','background-color':'#fff', 'border-top':'6px solid #007bff'}
                            ),#end of column div for total SME credit channeling
                        ], className="row flex-display"), #end of total channeling and NPL row div
                    
                    html.Div([#start of macroeconomic vars
                        html.H5("Indikator Makro Ekonomi", style={"font-weight":"bold"}),
                        html.Div([ #row div of macro vars
                            html.Div([ #pertumbuhan ekonomi div start
                                html.H5("Pertumbuhan Ekonomi", style={"font-weight":"bold", "color":"#fff"}),
                                dcc.Input(
                                    id="EconGrowth3",
                                    type="number",
                                    value=row_take['EconGrowth'].values[0],
                                    debounce=True
                                    ),
                                ],className="pretty_container four columns",style={"background-color":"#111"}),
                            html.Div([ #inflasi div start
                                html.H5("Tingkat Inflasi", style={"font-weight":"bold", "color":"#fff"}),
                                dcc.Input(
                                    id="Inflasi3",
                                    type="number",
                                    value=row_take['Inflasi'].values[0],
                                    debounce=True
                                    ),
                                ],className="pretty_container four columns",style={"background-color":"#111"}),
                            html.Div([ #pengangguran div start
                                html.H5("Tingkat Pengangguran", style={"font-weight":"bold", "color":"#fff"}),
                                dcc.Input(
                                    id="Unemployment3",
                                    type="number",
                                    value=row_take['Unemployment'].values[0],
                                    debounce=True
                                    ),
                                ],className="pretty_container four columns",style={"background-color":"#111"})
                            ],className="row flex-display") #end of macro vars row div
                        ],className="pretty_container"),#end of macroeconomic var div                    

                    html.Div([ #start of sectoral form div
                        html.Div([ #row div
                            html.Div([
                                html.H5("Nilai Penyaluran dan Proyeksi NPL Kredit UMKM per Sektor Ekonomi", style={"font-weight":"bold"}),                                
                                ], className="twelve columns"),
                            ],className="row flex-display"),#end of row div for title
                        #first row div for sectoral form
                        html.Div(children=[generate_form_eval_sector(i) for i in np.arange(3)
                            ],className="row flex-display",style={'width': '98%'}),
                        #second row of sectoral form
                        html.Div(children=[generate_form_eval_sector(i+3) for i in np.arange(3)
                            ],className="row flex-display",style={'width': '98%'}),
                        #third row of sectoral form
                        html.Div(children=[generate_form_eval_sector(i+6) for i in np.arange(3)
                            ],className="row flex-display",style={'width': '98%'}),
                        #fourth row of sectoral form
                        html.Div(children=[generate_form_eval_sector(i+9) for i in np.arange(3)
                            ],className="row flex-display",style={'width': '98%'}),
                        #fifth row of sectoral form
                        html.Div(children=[generate_form_eval_sector(i+12) for i in np.arange(3)
                            ],className="row flex-display",style={'width': '98%'}),
                        #sixth row of sectoral form
                        html.Div(children=[generate_form_eval_sector(i+15) for i in np.arange(3)
                            ],className="row flex-display",style={'width': '98%'}),
                        ],id="economic-sector-channeling-predictors3",
                        className="pretty_container",
                        style={"margin-bottom": "25px"}
                        ), #end of sectoral div form
                    
                    ]),#end of second tab
                #######################tab limit#####################
                dcc.Tab(label='Penganggaran IJP dan Loss Limit', children=[
                                        
                    html.Div([ #start of total channeling and NPL row div
                        html.Div([ #start of column div for total SME Credit channeling
                            html.H5("Total Penyaluran Kredit UMKM", style={'font-weight':'bold'}),
                            html.H4(id ="total_credit", style={'font-weight':'bold', 'font-size':'36px'}),
                            ], className="pretty_container six columns",
                            style={'text-align':'center','background-color':'#fff', 'border-top':'6px solid #fd7e14'}
                            ),#end of column div for total SME credit channeling
                        html.Div([ #start of column div for total NPL Projection
                            html.H5("Proyeksi Total NPL Kredit UMKM", style={'font-weight':'bold'}),
                            html.H1(id ="total_NPL", style={'font-weight':'bold', 'font-size':'44px'}),
                            html.P(id="total_NPL_val")
                            ], className="pretty_container six columns",
                            style={'text-align':'center','background-color':'#fff', 'border-top':'6px solid #ffc107'}
                            )#end of column div for total NPL projection
                        ], className="row flex-display"), #end of total channeling and NPL row div
                    
                    html.Div([ #start of budgeting row div
                        html.Div([ #start of column div for IJP tarif
                            html.H5("Tarif IJP Kredit UMKM", style={'font-weight':'bold'}),
                            html.H1(id ="IJP_tarif", style={'font-weight':'bold', 'font-size':'44px'}),
                            html.P(id="IJP_tarif_exp")
                            ], className="pretty_container four columns",
                            style={'text-align':'center','background-color':'#fff', 'border-top':'6px solid #007bff'}
                            ),#end of column div for IJP tarif
                        html.Div([ #start of column div for IJP budget
                            html.H5("Anggaran IJP", style={'font-weight':'bold'}),
                            html.H4(id ="IJP_budget", style={'font-weight':'bold', 'font-size':'32px'}),
                            ], className="pretty_container four columns",
                            style={'text-align':'center','background-color':'#fff', 'border-top':'6px solid #6610f2'}
                            ),#end of column div for IJP budget
                        html.Div([ #start of column div for loss limit budget
                            html.H5("Anggaran Loss Limit", style={'font-weight':'bold'}),
                            html.H4(id ="loss_limit_budget", style={'font-weight':'bold', 'font-size':'32px'}),
                            ], className="pretty_container four columns",
                            style={'text-align':'center','background-color':'#fff', 'border-top':'6px solid #6f42c1'}
                            ),#end of column div for loss limit budget
                        ], className="row flex-display"), #end of budgeting row div

                    html.Div([#start of macroeconomic vars
                        html.H5("Indikator Makro Ekonomi", style={"font-weight":"bold"}),
                        html.Div([ #row div of macro vars
                            html.Div([ #pertumbuhan ekonomi div start
                                html.H5("Pertumbuhan Ekonomi", style={"font-weight":"bold", "color":"#fff"}),
                                dcc.Input(
                                    id="EconGrowth",
                                    type="number",
                                    value=row_take['EconGrowth'].values[0],
                                    debounce=True
                                    ),
                                ],className="pretty_container four columns",style={"background-color":"#111"}),
                            html.Div([ #inflasi div start
                                html.H5("Tingkat Inflasi", style={"font-weight":"bold", "color":"#fff"}),
                                dcc.Input(
                                    id="Inflasi",
                                    type="number",
                                    value=row_take['Inflasi'].values[0],
                                    debounce=True
                                    ),
                                ],className="pretty_container four columns",style={"background-color":"#111"}),
                            html.Div([ #pengangguran div start
                                html.H5("Tingkat Pengangguran", style={"font-weight":"bold", "color":"#fff"}),
                                dcc.Input(
                                    id="Unemployment",
                                    type="number",
                                    value=row_take['Unemployment'].values[0],
                                    debounce=True
                                    ),
                                ],className="pretty_container four columns",style={"background-color":"#111"})
                            ],className="row flex-display") #end of macro vars row div
                        ],className="pretty_container"),#end of macroeconomic var div                    

                    html.Div([ #start of sectoral form div
                        html.Div([ #row div
                            html.Div([
                                html.H5("Nilai Penyaluran dan Proyeksi NPL Kredit UMKM per Sektor Ekonomi", style={"font-weight":"bold"}),                                
                                ], className="twelve columns"),
                            ],className="row flex-display"),#end of row div for title
                        #first row div for sectoral form
                        html.Div(children=[generate_form(i) for i in np.arange(3)
                            ],className="row flex-display",style={'width': '98%'}),
                        #second row of sectoral form
                        html.Div(children=[generate_form(i+3) for i in np.arange(3)
                            ],className="row flex-display",style={'width': '98%'}),
                        #third row of sectoral form
                        html.Div(children=[generate_form(i+6) for i in np.arange(3)
                            ],className="row flex-display",style={'width': '98%'}),
                        #fourth row of sectoral form
                        html.Div(children=[generate_form(i+9) for i in np.arange(3)
                            ],className="row flex-display",style={'width': '98%'}),
                        #fifth row of sectoral form
                        html.Div(children=[generate_form(i+12) for i in np.arange(3)
                            ],className="row flex-display",style={'width': '98%'}),
                        #sixth row of sectoral form
                        html.Div(children=[generate_form(i+15) for i in np.arange(3)
                            ],className="row flex-display",style={'width': '98%'}),
                        ],id="economic-sector-channeling-predictors",
                        className="pretty_container",
                        style={"margin-bottom": "25px"}
                        ), #end of sectoral div form

                    ]),#end of third tab
                #######################tab limit#####################
                dcc.Tab(label='Evaluasi Tarif IJP', children=[

                    html.Div([ #start of total channeling and NPL row div
                        html.Div([ #start of column div for total SME Credit channeling
                            html.H5("Total Penyaluran Kredit UMKM", style={'font-weight':'bold'}),
                            html.H4(id ="total_credit2", style={'font-weight':'bold', 'font-size':'28px'}),
                            ], className="pretty_container four columns",
                            style={'text-align':'center','background-color':'#fff', 'border-top':'6px solid #007bff'}
                            ),#end of column div for total SME credit channeling
                        html.Div([ #start of column div for total NPL Projection
                            html.H5("Proyeksi Total NPL Kredit UMKM", style={'font-weight':'bold'}),
                            html.H1(id ="total_NPL2", style={'font-weight':'bold', 'font-size':'44px'}),
                            html.P(id="total_NPL_val2")
                            ], className="pretty_container four columns",
                            style={'text-align':'center','background-color':'#fff', 'border-top':'6px solid #6610f2'}
                            ),#end of column div for total NPL projection
                        html.Div([ #start of column div for IJP tarif
                            html.H5("Tarif IJP Kredit UMKM", style={'font-weight':'bold'}),
                            html.H1(id ="IJP_tarif2", style={'font-weight':'bold', 'font-size':'44px'}),
                            html.P(id="IJP_tarif_exp2")
                            ], className="pretty_container four columns",
                            style={'text-align':'center','background-color':'#fff', 'border-top':'6px solid #6f42c1'}
                            )#end of column div for IJP tarif
                        ], className="row flex-display"), #end of total channeling and NPL row div
                    
                    html.Div([#start of macroeconomic vars
                        html.H5("Indikator Makro Ekonomi", style={"font-weight":"bold"}),
                        html.Div([ #row div of macro vars
                            html.Div([ #pertumbuhan ekonomi div start
                                html.H5("Pertumbuhan Ekonomi", style={"font-weight":"bold", "color":"#fff"}),
                                dcc.Input(
                                    id="EconGrowth2",
                                    type="number",
                                    value=row_take['EconGrowth'].values[0],
                                    debounce=True
                                    ),
                                ],className="pretty_container four columns",style={"background-color":"#111"}),
                            html.Div([ #inflasi div start
                                html.H5("Tingkat Inflasi", style={"font-weight":"bold", "color":"#fff"}),
                                dcc.Input(
                                    id="Inflasi2",
                                    type="number",
                                    value=row_take['Inflasi'].values[0],
                                    debounce=True
                                    ),
                                ],className="pretty_container four columns",style={"background-color":"#111"}),
                            html.Div([ #pengangguran div start
                                html.H5("Tingkat Pengangguran", style={"font-weight":"bold", "color":"#fff"}),
                                dcc.Input(
                                    id="Unemployment2",
                                    type="number",
                                    value=row_take['Unemployment'].values[0],
                                    debounce=True
                                    ),
                                ],className="pretty_container four columns",style={"background-color":"#111"})
                            ],className="row flex-display") #end of macro vars row div
                        ],className="pretty_container"),#end of macroeconomic var div                    

                    html.Div([ #start of sectoral form div
                        html.Div([ #row div
                            html.Div([
                                html.H5("Nilai Penyaluran dan Proyeksi NPL Kredit UMKM per Sektor Ekonomi", style={"font-weight":"bold"}),                                
                                ], className="twelve columns"),
                            ],className="row flex-display"),#end of row div for title
                        #first row div for sectoral form
                        html.Div(children=[generate_form_eval_IJP(i) for i in np.arange(3)
                            ],className="row flex-display",style={'width': '98%'}),
                        #second row of sectoral form
                        html.Div(children=[generate_form_eval_IJP(i+3) for i in np.arange(3)
                            ],className="row flex-display",style={'width': '98%'}),
                        #third row of sectoral form
                        html.Div(children=[generate_form_eval_IJP(i+6) for i in np.arange(3)
                            ],className="row flex-display",style={'width': '98%'}),
                        #fourth row of sectoral form
                        html.Div(children=[generate_form_eval_IJP(i+9) for i in np.arange(3)
                            ],className="row flex-display",style={'width': '98%'}),
                        #fifth row of sectoral form
                        html.Div(children=[generate_form_eval_IJP(i+12) for i in np.arange(3)
                            ],className="row flex-display",style={'width': '98%'}),
                        #sixth row of sectoral form
                        html.Div(children=[generate_form_eval_IJP(i+15) for i in np.arange(3)
                            ],className="row flex-display",style={'width': '98%'}),
                        ],id="economic-sector-channeling-predictors2",
                        className="pretty_container",
                        style={"margin-bottom": "25px"}
                        ), #end of sectoral div form
                    
                    ]),#end of fourth tab
                #######################tab limit#####################
                ])
            ]
        ), #end of main div
    html.Div([
        html.P("© 2020 - Inspektorat Jenderal Kementerian Keuangan", style={"font-weight":"bold"})
        ],className="pretty_container", style={'text-align':'center'})
        ]) #end of app.layout
    
@app.callback(
    Output('channel-graph-with-slider', 'figure'),
    [Input('year-slider', 'value'),Input('econ-sector-selector', 'value')])
def update_figure(selected_year,sektor):
    prefiltered_df = df[df.Tahun == selected_year]
    filtered_df = prefiltered_df[prefiltered_df.SektorEkonomi == sektor]

    fig = go.Figure()
    
    # Make traces for graph
    trace1 = go.Bar(x=monthCode, y=filtered_df['valueChannel'], xaxis='x2', yaxis='y2',
                marker=dict(color='#0099ff'),
                name='Penyaluran<br>Kredit')
    trace2 = go.Bar(x=monthCode, y=filtered_df['valueNPL'], xaxis='x2', yaxis='y2',
                marker=dict(color='#404040'),
                name='Non Performing<br>Loan')

    # Add trace data to figure
    fig.add_traces([trace1, trace2])


    # Update the margins to add a title and see graph x-labels.
    fig.layout.margin.update({'t':75, 'l':50})

    #legend setting
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
        ))

    #chart title and transition
    fig.layout.update({'title': 'Penyaluran Kredit dan NPL'})
    fig.update_layout(transition_duration=500)
    return fig

@app.callback(
    Output('channel-graph-with-slider-2', 'figure'),
    [Input('year-slider', 'value'),Input('econ-sector-selector', 'value')])
def update_figure2(selected_year,sektor):
    prefiltered_df = df[df.Tahun == selected_year]
    filtered_df = prefiltered_df[prefiltered_df.SektorEkonomi == sektor]
    
    fig = go.Figure(data=go.Scatter(x=monthCode, y=filtered_df['percentNPL']*100))

    #chart title and transition
    fig.layout.update({'title': 'Persentase NPL'})
    fig.update_layout(transition_duration=500)

    return fig

@app.callback(
    Output('aggregate-channel-graph-with-slider', 'figure'),
    Output('aggregate-npl-graph-with-slider', 'figure'),
    [Input('aggregate-year-slider', 'value')])
def update_aggregate2(selected_year):
    filtered_df = df[df.Tahun == selected_year]
    summ_df = filtered_df.groupby('Bulan', as_index=False).agg({"valueChannel":"sum"})
    summ_df2 = filtered_df.groupby('Bulan', as_index=False).agg({"valueNPL":"sum"})
    
    
    #initiate figure
    fig = go.Figure()
    fig2 = go.Figure()
    
    #data tracing
#    trace1 = go.Bar(x=monthCode, y=summ_df['valueChannel'], xaxis='x2', yaxis='y2',
#                marker=dict(color='#0099ff'),
#                name='Penyaluran<br>Kredit')
#    trace2 = go.Bar(x=monthCode, y=summ_df2['valueNPL'], xaxis='x2', yaxis='y2',
#                marker=dict(color='#6f42c1'),
#                name='Non Performing<br>Loan')
    
    trace1 = go.Scatter(x=monthCode, y=summ_df['valueChannel'], marker=dict(color='#0099ff'))
    trace2 = go.Scatter(x=monthCode, y=summ_df2['valueNPL'], marker=dict(color='#6f42c1'))
    
    # Add trace data to figure
    fig.add_traces([trace1])
    fig2.add_traces([trace2])

    #add title
    fig.layout.update({'title': 'Total Penyaluran Kredit'})
    fig2.layout.update({'title': 'Total NPL Kredit'})

    #chart transition
    fig.update_layout(transition_duration=500)
    fig2.update_layout(transition_duration=500)

    return fig, fig2

@app.callback(
    Output('channel-comparison-graph-with-slider-2', 'figure'),
    [Input('year-slider-3', 'value')])
def update_figure_comparison(selected_year):
    prefiltered_df = df[df.Tahun == selected_year]
    filtered_df = prefiltered_df[prefiltered_df.Bulan=="Jun"]

    fig = go.Figure(go.Bar(
            x=filtered_df['percentNPL']*100,
            y=econSector,
            orientation='h'))

    #chart title and transition
    fig.layout.update({'title': 'Perbandingan Persentase NPL Antar Sektor Ekonomi Tahun 2011-2020'})
    fig.update_layout(transition_duration=500)
    return fig

@app.callback(
    Output('channel-comparison-graph-with-slider', 'figure'),
    [Input('year-slider-2', 'value')])
def update_figure_comparison2(selected_year):
    prefiltered_df = df[df.Tahun == selected_year]
    filtered_df = prefiltered_df[prefiltered_df.Bulan=="Jun"]
    
    fig = px.pie(filtered_df, values='valueChannel', names='SektorEkonomi', color_discrete_sequence=px.colors.sequential.RdBu)

    #chart title and transition
    fig.layout.update({'title': 'Perbandingan Penyaluran Kredit Antar Sektor Ekonomi'})
    fig.update_layout(transition_duration=500)

    return fig

@app.callback(
    [Output("sector_NPL_{}".format(i), "children") for i in np.arange(18)],
    [Output("sector_NPL_val_{}".format(i), "children") for i in np.arange(18)],
    Output("total_NPL","children"),
    Output("total_NPL_val","children"),
    Output("total_credit","children"),
    Output("IJP_tarif","children"),
    Output("IJP_budget","children"),
    Output("loss_limit_budget","children"),
    [Input("EconGrowth", "value"),
    Input("Inflasi", "value"),
    Input("Unemployment", "value"),
    Input("sector_form_0", "value"),
    Input("sector_form_1", "value"),
    Input("sector_form_2", "value"),
    Input("sector_form_3", "value"),
    Input("sector_form_4", "value"),
    Input("sector_form_5", "value"),
    Input("sector_form_6", "value"),
    Input("sector_form_7", "value"),
    Input("sector_form_8", "value"),
    Input("sector_form_9", "value"),
    Input("sector_form_10", "value"),
    Input("sector_form_11", "value"),
    Input("sector_form_12", "value"),
    Input("sector_form_13", "value"),
    Input("sector_form_14", "value"),
    Input("sector_form_15", "value"),
    Input("sector_form_16", "value"),
    Input("sector_form_17", "value")
])
def predict_NPL(EconGrowth,Inflasi,Unemployment,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r):
    
    #prediction
    pred1 = model_rf.predict([[np.log(a),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]])
    pred2 = model_rf.predict([[np.log(b),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]])
    pred3 = model_rf.predict([[np.log(c),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]])
    pred4 = model_rf.predict([[np.log(d),1,Inflasi,EconGrowth,Unemployment,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    pred5 = model_rf.predict([[np.log(e),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]])
    pred6 = model_rf.predict([[np.log(f),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]])
    pred7 = model_rf.predict([[np.log(g),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]])
    pred8 = model_rf.predict([[np.log(h),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]])
    pred9 = model_rf.predict([[np.log(i),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]])
    pred10 = model_rf.predict([[np.log(j),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]])
    pred11 = model_rf.predict([[np.log(k),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]])
    pred12 = model_rf.predict([[np.log(l),1,Inflasi,EconGrowth,Unemployment,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    pred13 = model_rf.predict([[np.log(m),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]])
    pred14 = model_rf.predict([[np.log(n),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    pred15 = model_rf.predict([[np.log(o),1,Inflasi,EconGrowth,Unemployment,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    pred16 = model_rf.predict([[np.log(p),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]])
    pred17 = model_rf.predict([[np.log(q),1,Inflasi,EconGrowth,Unemployment,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    pred18 = model_rf.predict([[np.log(r),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]])
    
    pre = "Proyeksi NPL Kredit UMKM untuk sektor ekonomi "
    
    #processing sectoral NPL percentage
    preds1 = "{:,.2f} %".format(pred1[0]*100)
    preds2 = "{:,.2f} %".format(pred2[0]*100)
    preds3 = "{:,.2f} %".format(pred3[0]*100)
    preds4 = "{:,.2f} %".format(pred4[0]*100)
    preds5 = "{:,.2f} %".format(pred5[0]*100)
    preds6 = "{:,.2f} %".format(pred6[0]*100)
    preds7 = "{:,.2f} %".format(pred7[0]*100)
    preds8 = "{:,.2f} %".format(pred8[0]*100)
    preds9 = "{:,.2f} %".format(pred9[0]*100)
    preds10 = "{:,.2f} %".format(pred10[0]*100)
    preds11 = "{:,.2f} %".format(pred11[0]*100)
    preds12 = "{:,.2f} %".format(pred12[0]*100)
    preds13 = "{:,.2f} %".format(pred13[0]*100)
    preds14 = "{:,.2f} %".format(pred14[0]*100)
    preds15 = "{:,.2f} %".format(pred15[0]*100)
    preds16 = "{:,.2f} %".format(pred16[0]*100)
    preds17 = "{:,.2f} %".format(pred17[0]*100)
    preds18 = "{:,.2f} %".format(pred18[0]*100)
    
    #processing NPL sectoral value
    pref1 = pre+econSector[0]+" {:,.2f}".format(pred1[0]*a)
    pref2 = pre+econSector[1]+" {:,.2f}".format(pred2[0]*b)
    pref3 = pre+econSector[2]+" {:,.2f}".format(pred3[0]*c)
    pref4 = pre+econSector[3]+" {:,.2f}".format(pred4[0]*d)
    pref5 = pre+econSector[4]+" {:,.2f}".format(pred5[0]*e)
    pref6 = pre+econSector[5]+" {:,.2f}".format(pred6[0]*f)
    pref7 = pre+econSector[6]+" {:,.2f}".format(pred7[0]*g)
    pref8 = pre+econSector[7]+" {:,.2f}".format(pred8[0]*h)
    pref9 = pre+econSector[8]+" {:,.2f}".format(pred9[0]*i)
    pref10 = pre+econSector[9]+" {:,.2f}".format(pred10[0]*j)
    pref11 = pre+econSector[10]+" {:,.2f}".format(pred11[0]*k)
    pref12 = pre+econSector[11]+" {:,.2f}".format(pred12[0]*l)
    pref13 = pre+econSector[12]+" {:,.2f}".format(pred13[0]*m)
    pref14 = pre+econSector[13]+" {:,.2f}".format(pred14[0]*n)
    pref15 = pre+econSector[14]+" {:,.2f}".format(pred15[0]*o)
    pref16 = pre+econSector[15]+" {:,.2f}".format(pred16[0]*p)
    pref17 = pre+econSector[16]+" {:,.2f}".format(pred17[0]*q)
    pref18 = pre+econSector[17]+" {:,.2f}".format(pred18[0]*r)
    
    #populate variables
    credit_channeling = [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r]
    percent_NPL_prediction = [pred1[0],pred2[0],pred3[0],pred4[0],pred5[0],pred6[0],
                           pred7[0],pred8[0],pred9[0],pred10[0],pred11[0],pred12[0],
                           pred13[0],pred14[0],pred15[0],pred16[0],pred17[0],pred18[0]]
    
    #processing total NPL percentage and value
    total_SME_credit_channeling = sum(credit_channeling)
#    populate_total_NPL = [[credit_channeling[_]*percent_NPL_prediction[_]/100] for _ in np.arange(18)]
    total_NPL_val = 0.00
    for i in np.arange(18):
        NPL_i = credit_channeling[i]*percent_NPL_prediction[i]
        total_NPL_val +=NPL_i
    total_NPL_percentage = (total_NPL_val/total_SME_credit_channeling)*100
    
    #processing IJP and loss limit information
    #ijp_trf = total_NPL_percentage * 0.8 * 0.91
    ijp_trf = ((((total_NPL_percentage/100) * 0.8)-0.01) / 0.9)*100
    ijp = ijp_trf * total_SME_credit_channeling / 100
    #loss_lim = ijp / 100
    loss_lim = total_SME_credit_channeling / 100
    
    
    #data to pass
    total_NPL_percentage_pass = "{:,.2f} %".format(total_NPL_percentage)
    total_NPL_val_pass = "Proyeksi total nilai NPL atas Penyaluran Kredit kepada UMKM adalah Rp {:,.2f}".format(total_NPL_val)
    total_credit = "Rp {:,.2f}".format(total_SME_credit_channeling)
    ijp_tarif =  "{:,.2f} %".format(ijp_trf)
    ijp_budget = "Rp {:,.2f}".format(ijp)
    loss_limit_budget = "Rp {:,.2f}".format(loss_lim)
    
    return preds1, preds2, preds3, preds4, preds5, preds6, preds7, preds8, preds9, preds10, preds11, preds12, preds13, preds14, preds15, preds16, preds17, preds18, pref1, pref2, pref3, pref4, pref5, pref6, pref7, pref8, pref9, pref10, pref11, pref12, pref13, pref14, pref15, pref16, pref17, pref18, total_NPL_percentage_pass, total_NPL_val_pass, total_credit, ijp_tarif, ijp_budget, loss_limit_budget

######################limit second prediction######################

@app.callback(
    [Output("sector_NPL2_{}".format(i), "children") for i in np.arange(18)],
    [Output("sector_NPL_val2_{}".format(i), "children") for i in np.arange(18)],
    Output("total_NPL2","children"),
    Output("total_NPL_val2","children"),
    Output("total_credit2","children"),
    Output("IJP_tarif2","children"),
    [Input("EconGrowth2", "value"),
    Input("Inflasi2", "value"),
    Input("Unemployment2", "value"),
    Input("sector_form2_0", "value"),
    Input("sector_form2_1", "value"),
    Input("sector_form2_2", "value"),
    Input("sector_form2_3", "value"),
    Input("sector_form2_4", "value"),
    Input("sector_form2_5", "value"),
    Input("sector_form2_6", "value"),
    Input("sector_form2_7", "value"),
    Input("sector_form2_8", "value"),
    Input("sector_form2_9", "value"),
    Input("sector_form2_10", "value"),
    Input("sector_form2_11", "value"),
    Input("sector_form2_12", "value"),
    Input("sector_form2_13", "value"),
    Input("sector_form2_14", "value"),
    Input("sector_form2_15", "value"),
    Input("sector_form2_16", "value"),
    Input("sector_form2_17", "value")
])
def predict_NPL2(EconGrowth,Inflasi,Unemployment,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r):
    
    #prediction
    pred1 = model_rf.predict([[np.log(a),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]])
    pred2 = model_rf.predict([[np.log(b),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]])
    pred3 = model_rf.predict([[np.log(c),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]])
    pred4 = model_rf.predict([[np.log(d),1,Inflasi,EconGrowth,Unemployment,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    pred5 = model_rf.predict([[np.log(e),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]])
    pred6 = model_rf.predict([[np.log(f),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]])
    pred7 = model_rf.predict([[np.log(g),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]])
    pred8 = model_rf.predict([[np.log(h),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]])
    pred9 = model_rf.predict([[np.log(i),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]])
    pred10 = model_rf.predict([[np.log(j),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]])
    pred11 = model_rf.predict([[np.log(k),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]])
    pred12 = model_rf.predict([[np.log(l),1,Inflasi,EconGrowth,Unemployment,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    pred13 = model_rf.predict([[np.log(m),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]])
    pred14 = model_rf.predict([[np.log(n),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    pred15 = model_rf.predict([[np.log(o),1,Inflasi,EconGrowth,Unemployment,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    pred16 = model_rf.predict([[np.log(p),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]])
    pred17 = model_rf.predict([[np.log(q),1,Inflasi,EconGrowth,Unemployment,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    pred18 = model_rf.predict([[np.log(r),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]])
    
    pre = "Proyeksi NPL Kredit UMKM untuk sektor ekonomi "
    
    #processing sectoral NPL percentage
    preds1 = "{:,.2f} %".format(pred1[0]*100)
    preds2 = "{:,.2f} %".format(pred2[0]*100)
    preds3 = "{:,.2f} %".format(pred3[0]*100)
    preds4 = "{:,.2f} %".format(pred4[0]*100)
    preds5 = "{:,.2f} %".format(pred5[0]*100)
    preds6 = "{:,.2f} %".format(pred6[0]*100)
    preds7 = "{:,.2f} %".format(pred7[0]*100)
    preds8 = "{:,.2f} %".format(pred8[0]*100)
    preds9 = "{:,.2f} %".format(pred9[0]*100)
    preds10 = "{:,.2f} %".format(pred10[0]*100)
    preds11 = "{:,.2f} %".format(pred11[0]*100)
    preds12 = "{:,.2f} %".format(pred12[0]*100)
    preds13 = "{:,.2f} %".format(pred13[0]*100)
    preds14 = "{:,.2f} %".format(pred14[0]*100)
    preds15 = "{:,.2f} %".format(pred15[0]*100)
    preds16 = "{:,.2f} %".format(pred16[0]*100)
    preds17 = "{:,.2f} %".format(pred17[0]*100)
    preds18 = "{:,.2f} %".format(pred18[0]*100)
    
    #processing NPL sectoral value
    pref1 = pre+econSector[0]+" {:,.2f}".format(pred1[0]*a)
    pref2 = pre+econSector[1]+" {:,.2f}".format(pred2[0]*b)
    pref3 = pre+econSector[2]+" {:,.2f}".format(pred3[0]*c)
    pref4 = pre+econSector[3]+" {:,.2f}".format(pred4[0]*d)
    pref5 = pre+econSector[4]+" {:,.2f}".format(pred5[0]*e)
    pref6 = pre+econSector[5]+" {:,.2f}".format(pred6[0]*f)
    pref7 = pre+econSector[6]+" {:,.2f}".format(pred7[0]*g)
    pref8 = pre+econSector[7]+" {:,.2f}".format(pred8[0]*h)
    pref9 = pre+econSector[8]+" {:,.2f}".format(pred9[0]*i)
    pref10 = pre+econSector[9]+" {:,.2f}".format(pred10[0]*j)
    pref11 = pre+econSector[10]+" {:,.2f}".format(pred11[0]*k)
    pref12 = pre+econSector[11]+" {:,.2f}".format(pred12[0]*l)
    pref13 = pre+econSector[12]+" {:,.2f}".format(pred13[0]*m)
    pref14 = pre+econSector[13]+" {:,.2f}".format(pred14[0]*n)
    pref15 = pre+econSector[14]+" {:,.2f}".format(pred15[0]*o)
    pref16 = pre+econSector[15]+" {:,.2f}".format(pred16[0]*p)
    pref17 = pre+econSector[16]+" {:,.2f}".format(pred17[0]*q)
    pref18 = pre+econSector[17]+" {:,.2f}".format(pred18[0]*r)
    
    #populate variables
    credit_channeling = [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r]
    percent_NPL_prediction = [pred1[0],pred2[0],pred3[0],pred4[0],pred5[0],pred6[0],
                           pred7[0],pred8[0],pred9[0],pred10[0],pred11[0],pred12[0],
                           pred13[0],pred14[0],pred15[0],pred16[0],pred17[0],pred18[0]]
    
    #processing total NPL percentage and value
    total_SME_credit_channeling = sum(credit_channeling)
#    populate_total_NPL = [[credit_channeling[_]*percent_NPL_prediction[_]/100] for _ in np.arange(18)]
    total_NPL_val = 0.00
    for i in np.arange(18):
        NPL_i = credit_channeling[i]*percent_NPL_prediction[i]
        total_NPL_val +=NPL_i
    total_NPL_percentage = (total_NPL_val/total_SME_credit_channeling)*100
    
    #processing IJP and loss limit information
    #ijp_trf = total_NPL_percentage * 0.8 * 0.91
    ijp_trf = ((((total_NPL_percentage/100) * 0.8)-0.01) / 0.9)*100
    
    #data to pass
    total_NPL_percentage_pass = "{:,.2f} %".format(total_NPL_percentage)
    total_NPL_val_pass = "Proyeksi total nilai NPL atas Penyaluran Kredit kepada UMKM adalah Rp {:,.2f}".format(total_NPL_val)
    total_credit = "Rp {:,.2f}".format(total_SME_credit_channeling)
    ijp_tarif =  "{:,.2f} %".format(ijp_trf)
    
    return preds1, preds2, preds3, preds4, preds5, preds6, preds7, preds8, preds9, preds10, preds11, preds12, preds13, preds14, preds15, preds16, preds17, preds18, pref1, pref2, pref3, pref4, pref5, pref6, pref7, pref8, pref9, pref10, pref11, pref12, pref13, pref14, pref15, pref16, pref17, pref18, total_NPL_percentage_pass, total_NPL_val_pass, total_credit, ijp_tarif

######################limit third prediction######################

@app.callback(
    [Output("sector_NPL3_{}".format(i), "children") for i in np.arange(18)],
    [Output("sector_NPL_val3_{}".format(i), "children") for i in np.arange(18)],
    [Output("sector_compare_NPL3_{}".format(i), "children") for i in np.arange(18)],
    Output("channel-comparison-graph-sector-affected","figure"),
    [Input("EconGrowth3", "value"),
    Input("Inflasi3", "value"),
    Input("Unemployment3", "value"),
    Input("npl_value_type", "value"),
    Input("sector_form3_0", "value"),
    Input("sector_form3_1", "value"),
    Input("sector_form3_2", "value"),
    Input("sector_form3_3", "value"),
    Input("sector_form3_4", "value"),
    Input("sector_form3_5", "value"),
    Input("sector_form3_6", "value"),
    Input("sector_form3_7", "value"),
    Input("sector_form3_8", "value"),
    Input("sector_form3_9", "value"),
    Input("sector_form3_10", "value"),
    Input("sector_form3_11", "value"),
    Input("sector_form3_12", "value"),
    Input("sector_form3_13", "value"),
    Input("sector_form3_14", "value"),
    Input("sector_form3_15", "value"),
    Input("sector_form3_16", "value"),
    Input("sector_form3_17", "value")
])
def predict_NPL2(EconGrowth,Inflasi,Unemployment,val_type,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r):
    
    #prediction
    pred1 = model_rf.predict([[np.log(a),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]])
    pred2 = model_rf.predict([[np.log(b),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]])
    pred3 = model_rf.predict([[np.log(c),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]])
    pred4 = model_rf.predict([[np.log(d),1,Inflasi,EconGrowth,Unemployment,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    pred5 = model_rf.predict([[np.log(e),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]])
    pred6 = model_rf.predict([[np.log(f),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]])
    pred7 = model_rf.predict([[np.log(g),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]])
    pred8 = model_rf.predict([[np.log(h),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]])
    pred9 = model_rf.predict([[np.log(i),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]])
    pred10 = model_rf.predict([[np.log(j),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]])
    pred11 = model_rf.predict([[np.log(k),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]])
    pred12 = model_rf.predict([[np.log(l),1,Inflasi,EconGrowth,Unemployment,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    pred13 = model_rf.predict([[np.log(m),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]])
    pred14 = model_rf.predict([[np.log(n),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    pred15 = model_rf.predict([[np.log(o),1,Inflasi,EconGrowth,Unemployment,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    pred16 = model_rf.predict([[np.log(p),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]])
    pred17 = model_rf.predict([[np.log(q),1,Inflasi,EconGrowth,Unemployment,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    pred18 = model_rf.predict([[np.log(r),1,Inflasi,EconGrowth,Unemployment,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]])
    
    pre = "Proyeksi NPL Kredit UMKM untuk sektor ekonomi "
    
    #processing sectoral NPL percentage
    preds1 = "{:,.2f} %".format(pred1[0]*100)
    preds2 = "{:,.2f} %".format(pred2[0]*100)
    preds3 = "{:,.2f} %".format(pred3[0]*100)
    preds4 = "{:,.2f} %".format(pred4[0]*100)
    preds5 = "{:,.2f} %".format(pred5[0]*100)
    preds6 = "{:,.2f} %".format(pred6[0]*100)
    preds7 = "{:,.2f} %".format(pred7[0]*100)
    preds8 = "{:,.2f} %".format(pred8[0]*100)
    preds9 = "{:,.2f} %".format(pred9[0]*100)
    preds10 = "{:,.2f} %".format(pred10[0]*100)
    preds11 = "{:,.2f} %".format(pred11[0]*100)
    preds12 = "{:,.2f} %".format(pred12[0]*100)
    preds13 = "{:,.2f} %".format(pred13[0]*100)
    preds14 = "{:,.2f} %".format(pred14[0]*100)
    preds15 = "{:,.2f} %".format(pred15[0]*100)
    preds16 = "{:,.2f} %".format(pred16[0]*100)
    preds17 = "{:,.2f} %".format(pred17[0]*100)
    preds18 = "{:,.2f} %".format(pred18[0]*100)
    
    #processing NPL sectoral value
    pref1 = pre+econSector[0]+" {:,.2f}".format(pred1[0]*a)
    pref2 = pre+econSector[1]+" {:,.2f}".format(pred2[0]*b)
    pref3 = pre+econSector[2]+" {:,.2f}".format(pred3[0]*c)
    pref4 = pre+econSector[3]+" {:,.2f}".format(pred4[0]*d)
    pref5 = pre+econSector[4]+" {:,.2f}".format(pred5[0]*e)
    pref6 = pre+econSector[5]+" {:,.2f}".format(pred6[0]*f)
    pref7 = pre+econSector[6]+" {:,.2f}".format(pred7[0]*g)
    pref8 = pre+econSector[7]+" {:,.2f}".format(pred8[0]*h)
    pref9 = pre+econSector[8]+" {:,.2f}".format(pred9[0]*i)
    pref10 = pre+econSector[9]+" {:,.2f}".format(pred10[0]*j)
    pref11 = pre+econSector[10]+" {:,.2f}".format(pred11[0]*k)
    pref12 = pre+econSector[11]+" {:,.2f}".format(pred12[0]*l)
    pref13 = pre+econSector[12]+" {:,.2f}".format(pred13[0]*m)
    pref14 = pre+econSector[13]+" {:,.2f}".format(pred14[0]*n)
    pref15 = pre+econSector[14]+" {:,.2f}".format(pred15[0]*o)
    pref16 = pre+econSector[15]+" {:,.2f}".format(pred16[0]*p)
    pref17 = pre+econSector[16]+" {:,.2f}".format(pred17[0]*q)
    pref18 = pre+econSector[17]+" {:,.2f}".format(pred18[0]*r)
    
    #populate variables
    credit_channeling = [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r]
    percent_NPL_prediction = [pred1[0],pred2[0],pred3[0],pred4[0],pred5[0],pred6[0],
                           pred7[0],pred8[0],pred9[0],pred10[0],pred11[0],pred12[0],
                           pred13[0],pred14[0],pred15[0],pred16[0],pred17[0],pred18[0]]
    
    s1 = pd.Series(econSector, name='sector')
    s2 = pd.Series(credit_channeling, name='channeling')
    s3 = pd.Series(percent_NPL_prediction, name='percentNPL')
    
    df = pd.concat([s1,s2,s3], axis=1)
    df['valueNPL'] = df['percentNPL']*df['channeling']    
    
    fig = go.Figure(go.Bar(
            x=df['percentNPL']*100 if val_type=='Percentage' else df['valueNPL'],
            y=econSector,
            orientation='h'))

    #chart transition
    fig.update_layout(transition_duration=500)
    
    avg_word_2019_0 = "{:,.2f} %".format(avg_2019[0])
    avg_word_2019_1 = "{:,.2f} %".format(avg_2019[1])
    avg_word_2019_2 = "{:,.2f} %".format(avg_2019[2])
    avg_word_2019_3 = "{:,.2f} %".format(avg_2019[3])
    avg_word_2019_4 = "{:,.2f} %".format(avg_2019[4])
    avg_word_2019_5 = "{:,.2f} %".format(avg_2019[5])
    avg_word_2019_6 = "{:,.2f} %".format(avg_2019[6])
    avg_word_2019_7 = "{:,.2f} %".format(avg_2019[7])
    avg_word_2019_8 = "{:,.2f} %".format(avg_2019[8])
    avg_word_2019_9 = "{:,.2f} %".format(avg_2019[9])    
    avg_word_2019_10 = "{:,.2f} %".format(avg_2019[10])
    avg_word_2019_11 = "{:,.2f} %".format(avg_2019[11])
    avg_word_2019_12 = "{:,.2f} %".format(avg_2019[12])
    avg_word_2019_13 = "{:,.2f} %".format(avg_2019[13])
    avg_word_2019_14 = "{:,.2f} %".format(avg_2019[14])
    avg_word_2019_15 = "{:,.2f} %".format(avg_2019[15])
    avg_word_2019_16 = "{:,.2f} %".format(avg_2019[16])
    avg_word_2019_17 = "{:,.2f} %".format(avg_2019[17])
    
    return preds1, preds2, preds3, preds4, preds5, preds6, preds7, preds8, preds9, preds10, preds11, preds12, preds13, preds14, preds15, preds16, preds17, preds18, pref1, pref2, pref3, pref4, pref5, pref6, pref7, pref8, pref9, pref10, pref11, pref12, pref13, pref14, pref15, pref16, pref17, pref18, avg_word_2019_0, avg_word_2019_1, avg_word_2019_2, avg_word_2019_3, avg_word_2019_4, avg_word_2019_5, avg_word_2019_6, avg_word_2019_7, avg_word_2019_8, avg_word_2019_9, avg_word_2019_10, avg_word_2019_11, avg_word_2019_12, avg_word_2019_13, avg_word_2019_14, avg_word_2019_15, avg_word_2019_16, avg_word_2019_17, fig

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)

