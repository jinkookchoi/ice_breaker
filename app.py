from flask import Flask, render_template, request, jsonify, Response
from dotenv import load_dotenv

from ice_breaker import ice_break_with

load_dotenv()

app = Flask(__name__)


@app.route("/")
def index() -> str:
    html: str = render_template("index.html")
    return html


@app.route("/process", methods=["POST"])
def process() -> Response:
    name = request.form["name"]

    # summary_and_facts, interests, ice_breakers, profile_pic_url = ice_break_with(
    summary, profile_pic_url = ice_break_with(name=name)

    return jsonify(
        {
            "summary_and_facts": summary.to_dict(),
            # "interests": interests.to_dict(),
            # "ice_breakers": ice_breakers.to_dict(),
            "picture_url": profile_pic_url,
        }
    )


if __name__ == "__main__":

    app.run(host="0.0.0.0", debug=True, port=8765)
