from flask import Flask, render_template, request, redirect, url_for
import plotly.express as px
from random import randint

app = Flask(__name__)


@app.route('/')
def main():
    return render_template("index.html")

@app.route('/graph')
def graph():
    s = randint(10,30)
    fig =px.scatter(x=range(s), y=range(s))
    fig.write_html("templates/gen/graph.html")
    return redirect(url_for('main'))

if __name__ == "__main__":
    app.run(debug = True)
