import dash
import pandas as pd
import sqlite3
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

try:
    conexao = sqlite3.connect('base_colaboradores.db')
    df = pd.read_sql_query('SELECT * FROM colaboradores', conexao)
except:
    print('Base de dados pronta para uso')

sidebar = html.Div([
    html.Div([html.H5('Menu Principal')]),
    dbc.Nav([
        dbc.NavLink('Cadastro de Colaboradores',
                    href = '/cadastro',
                    active = 'exact'),
        dbc.NavLink('Lista de Colaboradores',
                    href = '/lista',
                    active = 'exact')],
            vertical = True,
            pills = True)],
                id = 'sidebar',
                style = {'position': 'fixed',
                         'top': 0,
                         'left': 0,
                         'bottom': 0,
                         'width': 200,
                         'padding': 5,
                         'background-color': '#f8f9fa'})

conteudo = html.Div(id = 'conteudo_pagina',
                    style = {'margin-left':'18rem',
                             'margin-right':'2rem',
                             'padding':'2rem 1rem'})

pag_lista = html.Div([
    html.H4('Lista de Colaboradores'),
    html.P(),
    dcc.Dropdown(id = 'seleciona_colaborador',
        options = [
            {'label':linha['nome'], 'value': indice} for indice, linha in df.iterrows()],
        value = None),
    html.P(),
    html.Div(id = 'saida_lista_colaboradores')
], id = 'pagina-1')

pag_cadastro = html.Div([
    html.H4('Cadastro de Colaboradores'),
    dbc.Row([
        dbc.Label('Nome Completo: ', width = 4),
        dbc.Col(dbc.Input(id = 'entrada_nome', type = 'text'), width = 8)]),
    dbc.Row([
        dbc.Label('Identidade: ', width = 4),
        dbc.Col(dbc.Input(id = 'entrada_identidade', type = 'text'), width = 8)]),
    dbc.Row([
        dbc.Label('CPF: ', width = 4),
        dbc.Col(dbc.Input(id = 'entrada_cpf', type = 'text'), width = 8)]),
    dbc.Row([
        dbc.Label('Telefone: ', width = 4),
        dbc.Col(dbc.Input(id = 'entrada_telefone', type = 'text'), width = 8)]),
    dbc.Row([
        dbc.Label('Endereço: ', width = 4),
        dbc.Col(dbc.Input(id = 'entrada_endereco', type = 'text'), width = 8)]),
    dbc.Row([
        dbc.Label('Foto: ', width = 4),
        dcc.Upload(id = 'entrada_foto', children = html.Div([
            'Arraste e solte ou ',
            html.A('clique aqui para selecionar um arquivo.')]),
        style = {'borderWidth':'2px',
                 'borderStyle':'dashed',
                 'borderRadius':'6px',
                 'textAlign':'center'})]),
    html.P(),
    dbc.Button('Cadastrar', id = 'botao_cadastrar', color = 'primary', n_clicks = 0),
    html.P(),
    html.Div(id = 'saida_formulario')
], id = 'pagina-2')

app.layout = html.Div([dcc.Location(id = 'url'),
                       sidebar,
                       conteudo])

@app.callback(Output('conteudo_pagina', 'children'),
              [Input('url', 'pathname')])
def carrega_pagina(e):
    if e == '/cadastro':
        return pag_cadastro
    elif e == '/lista':
        return pag_lista
    return html.P('Página não encontrada')

@app.callback(Output('saida_formulario', 'children'),
              [Input('botao_cadastrar', 'n_clicks')],
              [State('entrada_nome', 'value'),
               State('entrada_identidade', 'value'),
               State('entrada_cpf', 'value'),
               State('entrada_telefone', 'value'),
               State('entrada_endereco', 'value'),
               State('entrada_foto', 'contents')])
def cadastra(n_clicks, nome, identidade, cpf, telefone, endereco, foto):
    if not foto:
        return html.Div('Nenhuma imagem carregada')
    conexao = sqlite3.connect('base_colaboradores.db')
    cursor = conexao.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS colaboradores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT,
            identidade TEXT,
            cpf TEXT,
            telefone TEXT,
            endereco TEXT,
            foto BLOB)""")
    dados_foto = foto.encode()
    cursor.execute("""
        INSERT INTO colaboradores (nome, identidade, cpf, telefone, endereco, foto)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (nome, identidade, cpf, telefone, endereco, dados_foto))
    conexao.commit()
    conexao.close()
    return html.Div('Colaborador cadastrado com sucesso!!!')

@app.callback(Output('saida_lista_colaboradores', 'children'),
              [Input('seleciona_colaborador', 'value')])
def exibe_dados_colaborador(col):
    if col is None:
        return []
    linha = df.iloc[col]
    return html.Div([
        html.Img(src = linha['foto'].decode(), style = {'width':'80px', 'height':'80px'}),
        html.P(f'Nome: {linha["nome"]}'),
        html.P(f'Identidade: {linha["identidade"]}'),
        html.P(f'CPF: {linha["cpf"]}'),
        html.P(f'Telefone: {linha["telefone"]}'),
        html.P(f'Endereço: {linha["endereco"]}')])

conexao.close()

app.run()
