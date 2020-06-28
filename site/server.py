from flask import Flask, render_template, request, redirect, url_for
import plotly.express as px
from random import randint

app = Flask(__name__)


@app.route('/')
def main():
    return render_template("index.html", type = "-")

@app.route('/first_analysis')
def mitya():
    return render_template("mitya.html", type = "mitya")

@app.route('/graph')
def graph():
    s = randint(10,30)
    fig =px.scatter(x=range(s), y=range(s), height=300)
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="#f8f9fa",
    )
    fig.write_html("templates/main_page/empty.html")
    fig.write_html("templates/main_page/empty1.html")
    fig.write_html("templates/main_page/empty2.html")
    return redirect(url_for('main'))

@app.route('/search')
def search_get():
    id = request.args.get('id')
    if not id:
        return redirect(url_for('main'))
    return render_template("search.html", id = id, type = "-")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug = True)
