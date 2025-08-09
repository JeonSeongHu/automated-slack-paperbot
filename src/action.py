from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content

from datetime import date, datetime

import argparse
import yaml
import os
from dotenv import load_dotenv
import openai
from bs4 import BeautifulSoup
from slack_sdk import WebClient
from relevancy import generate_relevance_score, process_subject_fields
from download_new_papers import get_papers


# Hackathon quality code. Don't judge too harshly.
# Feel free to submit pull requests to improve the code.

topics = {
    "Physics": "",
    "Mathematics": "math",
    "Computer Science": "cs",
    "Quantitative Biology": "q-bio",
    "Quantitative Finance": "q-fin",
    "Statistics": "stat",
    "Electrical Engineering and Systems Science": "eess",
    "Economics": "econ",
}

physics_topics = {
    "Astrophysics": "astro-ph",
    "Condensed Matter": "cond-mat",
    "General Relativity and Quantum Cosmology": "gr-qc",
    "High Energy Physics - Experiment": "hep-ex",
    "High Energy Physics - Lattice": "hep-lat",
    "High Energy Physics - Phenomenology": "hep-ph",
    "High Energy Physics - Theory": "hep-th",
    "Mathematical Physics": "math-ph",
    "Nonlinear Sciences": "nlin",
    "Nuclear Experiment": "nucl-ex",
    "Nuclear Theory": "nucl-th",
    "Physics": "physics",
    "Quantum Physics": "quant-ph",
}


# TODO: surely theres a better way
category_map = {
    "Astrophysics": [
        "Astrophysics of Galaxies",
        "Cosmology and Nongalactic Astrophysics",
        "Earth and Planetary Astrophysics",
        "High Energy Astrophysical Phenomena",
        "Instrumentation and Methods for Astrophysics",
        "Solar and Stellar Astrophysics",
    ],
    "Condensed Matter": [
        "Disordered Systems and Neural Networks",
        "Materials Science",
        "Mesoscale and Nanoscale Physics",
        "Other Condensed Matter",
        "Quantum Gases",
        "Soft Condensed Matter",
        "Statistical Mechanics",
        "Strongly Correlated Electrons",
        "Superconductivity",
    ],
    "General Relativity and Quantum Cosmology": ["None"],
    "High Energy Physics - Experiment": ["None"],
    "High Energy Physics - Lattice": ["None"],
    "High Energy Physics - Phenomenology": ["None"],
    "High Energy Physics - Theory": ["None"],
    "Mathematical Physics": ["None"],
    "Nonlinear Sciences": [
        "Adaptation and Self-Organizing Systems",
        "Cellular Automata and Lattice Gases",
        "Chaotic Dynamics",
        "Exactly Solvable and Integrable Systems",
        "Pattern Formation and Solitons",
    ],
    "Nuclear Experiment": ["None"],
    "Nuclear Theory": ["None"],
    "Physics": [
        "Accelerator Physics",
        "Applied Physics",
        "Atmospheric and Oceanic Physics",
        "Atomic and Molecular Clusters",
        "Atomic Physics",
        "Biological Physics",
        "Chemical Physics",
        "Classical Physics",
        "Computational Physics",
        "Data Analysis, Statistics and Probability",
        "Fluid Dynamics",
        "General Physics",
        "Geophysics",
        "History and Philosophy of Physics",
        "Instrumentation and Detectors",
        "Medical Physics",
        "Optics",
        "Physics and Society",
        "Physics Education",
        "Plasma Physics",
        "Popular Physics",
        "Space Physics",
    ],
    "Quantum Physics": ["None"],
    "Mathematics": [
        "Algebraic Geometry",
        "Algebraic Topology",
        "Analysis of PDEs",
        "Category Theory",
        "Classical Analysis and ODEs",
        "Combinatorics",
        "Commutative Algebra",
        "Complex Variables",
        "Differential Geometry",
        "Dynamical Systems",
        "Functional Analysis",
        "General Mathematics",
        "General Topology",
        "Geometric Topology",
        "Group Theory",
        "History and Overview",
        "Information Theory",
        "K-Theory and Homology",
        "Logic",
        "Mathematical Physics",
        "Metric Geometry",
        "Number Theory",
        "Numerical Analysis",
        "Operator Algebras",
        "Optimization and Control",
        "Probability",
        "Quantum Algebra",
        "Representation Theory",
        "Rings and Algebras",
        "Spectral Theory",
        "Statistics Theory",
        "Symplectic Geometry",
    ],
    "Computer Science": [
        "Artificial Intelligence",
        "Computation and Language",
        "Computational Complexity",
        "Computational Engineering, Finance, and Science",
        "Computational Geometry",
        "Computer Science and Game Theory",
        "Computer Vision and Pattern Recognition",
        "Computers and Society",
        "Cryptography and Security",
        "Data Structures and Algorithms",
        "Databases",
        "Digital Libraries",
        "Discrete Mathematics",
        "Distributed, Parallel, and Cluster Computing",
        "Emerging Technologies",
        "Formal Languages and Automata Theory",
        "General Literature",
        "Graphics",
        "Hardware Architecture",
        "Human-Computer Interaction",
        "Information Retrieval",
        "Information Theory",
        "Logic in Computer Science",
        "Machine Learning",
        "Mathematical Software",
        "Multiagent Systems",
        "Multimedia",
        "Networking and Internet Architecture",
        "Neural and Evolutionary Computing",
        "Numerical Analysis",
        "Operating Systems",
        "Other Computer Science",
        "Performance",
        "Programming Languages",
        "Robotics",
        "Social and Information Networks",
        "Software Engineering",
        "Sound",
        "Symbolic Computation",
        "Systems and Control",
    ],
    "Quantitative Biology": [
        "Biomolecules",
        "Cell Behavior",
        "Genomics",
        "Molecular Networks",
        "Neurons and Cognition",
        "Other Quantitative Biology",
        "Populations and Evolution",
        "Quantitative Methods",
        "Subcellular Processes",
        "Tissues and Organs",
    ],
    "Quantitative Finance": [
        "Computational Finance",
        "Economics",
        "General Finance",
        "Mathematical Finance",
        "Portfolio Management",
        "Pricing of Securities",
        "Risk Management",
        "Statistical Finance",
        "Trading and Market Microstructure",
    ],
    "Statistics": [
        "Applications",
        "Computation",
        "Machine Learning",
        "Methodology",
        "Other Statistics",
        "Statistics Theory",
    ],
    "Electrical Engineering and Systems Science": [
        "Audio and Speech Processing",
        "Image and Video Processing",
        "Signal Processing",
        "Systems and Control",
    ],
    "Economics": ["Econometrics", "General Economics", "Theoretical Economics"],
}


def generate_body(topic, categories, interest, threshold, config_path="config.yaml"):
    if topic == "Physics":
        raise RuntimeError("You must choose a physics subtopic.")
    elif topic in physics_topics:
        abbr = physics_topics[topic]
    elif topic in topics:
        abbr = topics[topic]
    else:
        raise RuntimeError(f"Invalid topic {topic}")
    if categories:
        for category in categories:
            if category not in category_map[topic]:
                raise RuntimeError(f"{category} is not a category of {topic}")
        papers = get_papers(abbr)
        papers = [
            t
            for t in papers
            if bool(set(process_subject_fields(t["subjects"])) & set(categories))
        ]
    else:
        papers = get_papers(abbr)
    if interest:
        relevancy, hallucination = generate_relevance_score(
            papers,
            query={"interest": interest},
            threshold_score=threshold,
            num_paper_in_prompt=16,
            config_path=config_path,
        )
        body = "<br><br>".join(
            [
                f'Title: <a href="{paper["main_page"]}">{paper["title"]}</a><br>Authors: {paper["authors"]}<br>Score: {paper["Relevancy score"]}<br>Reason: {paper["Reasons for match"]}'
                for paper in relevancy
            ]
        )
        if hallucination:
            body = (
                "Warning: the model hallucinated some papers. We have tried to remove them, but the scores may not be accurate.<br><br>"
                + body
            )
        return body, relevancy, hallucination
    else:
        body = "<br><br>".join(
            [
                f'Title: <a href="{paper["main_page"]}">{paper["title"]}</a><br>Authors: {paper["authors"]}'
                for paper in papers
            ]
        )
        return body, [], False


if __name__ == "__main__":
    # Load the .env file.
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="yaml config file to use", default="config.yaml"
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("No openai api key found")
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    topic = config["topic"]
    categories = config["categories"]
    from_email = os.environ.get("FROM_EMAIL")
    to_email = os.environ.get("TO_EMAIL")
    threshold = config["threshold"]
    interest = config["interest"]
    body, selected_list, hallucination = generate_body(topic, categories, interest, threshold, args.config)
    # Fallback: if body is empty but we have selected_list, synthesize HTML
    if (not body or body.strip() == "") and selected_list:
        rows = []
        for p in selected_list:
            rows.append(
                f'<div><b><a href="{p.get("main_page", "#")}">{p.get("title", "")}</a></b><br/>'
                f'Authors: {p.get("authors", "")}<br/>'
                f'Score: {p.get("Relevancy score", "?")} / Novelty: {p.get("Novelty score", "?")} / Priority: {p.get("Priority", "")}<br/>'
                f'Reason: {p.get("Reasons for match", "")}<br/></div><hr/>'
            )
        body = "\n".join(rows)
    with open("digest.html", "w") as f:
        f.write(body or "")
    # Save structured selection for downstream usage
    if selected_list:
        import json
        with open("digest.json", "w") as jf:
            json.dump(selected_list, jf)
    # Optional Slack notification via Slack Web API (Bot Token)
    slack_bot_token = os.environ.get("SLACK_BOT_TOKEN")
    slack_channel = os.environ.get("SLACK_CHANNEL")
    if slack_bot_token and slack_channel and selected_list:
        # Sort by rating desc, plain-text summary message (no Block Kit)
        sorted_items = sorted(selected_list, key=lambda x: int(x.get("Relevancy score", 0)), reverse=True)
        top_k = int(os.environ.get("SLACK_TOP_K", "30"))
        picked = sorted_items[:top_k]

        def _t(text: str, limit: int) -> str:
            if not text:
                return ""
            s = text.strip().replace("\n", " ")
            return s if len(s) <= limit else s[: limit - 1] + "…"

        # Format header date in KST to match arXiv KST scraping window
        import pytz
        kst_day = datetime.now(tz=pytz.timezone("Asia/Seoul")).strftime('%Y-%m-%d')
        lines = []
        lines.append(f"arXiv Digest (KST) - {kst_day}")
        lines.append("Focus: 3D Vision, Diffusion, Deep Learning, Computer Vision")
        lines.append(f"Items: {len(picked)} (sorted by rating)")
        lines.append("")
        for idx, paper in enumerate(picked, start=1):
            title = paper.get("title", "").strip()
            link = paper.get("main_page", "").strip()
            authors = paper.get("authors", "").strip()
            authors_short = ", ".join([a.strip() for a in authors.split(",")][:3])
            rel = paper.get("Relevancy score", "?")
            nov = paper.get("Novelty score", "?")
            pri = paper.get("Priority", "Skim")
            # Always prefer Korean reasons; if English, 그대로 출력되지만 프롬프트로 한국어를 강제해 둠
            reason = _t(paper.get("Reasons for match", ""), 220)
            abstract = _t(paper.get("abstract", ""), 220)
            venue = paper.get("Venue", "")
            # Use robustly-scraped project_url only (avoid LLM hallucinated 'Project page')
            proj_raw = paper.get("project_url", "")
            # Clean trailing punctuation from URL if any
            proj = proj_raw.rstrip('.,);]\'"') if proj_raw else ""
            lines.append(f"{idx}. <{link}|{title}>")
            lines.append(f"   R {rel}/10 | N {nov}/10 | P {pri} | {authors_short}")
            if venue:
                lines.append(f"   Venue: {venue}")
            if proj:
                lines.append(f"   Project: {proj}")
            lines.append(f"   이유: {reason}")
            lines.append(f"   Abs: {abstract}")
            lines.append("")

        client = WebClient(token=slack_bot_token)
        resp = client.chat_postMessage(channel=slack_channel, text="\n".join(lines))
        ok = resp.get("ok", False)
        ch = resp.get("channel", "")
        ts = resp.get("ts", "")
        print(f"Slack notification: ok={ok} channel={ch} ts={ts}")
    if os.environ.get("SENDGRID_API_KEY", None):
        sg = SendGridAPIClient(api_key=os.environ.get("SENDGRID_API_KEY"))
        from_email = Email(from_email)  # Change to your verified sender
        to_email = To(to_email)
        subject = date.today().strftime("Personalized arXiv Digest, %d %b %Y")
        content = Content("text/html", body)
        mail = Mail(from_email, to_email, subject, content)
        mail_json = mail.get()

        # Send an HTTP POST request to /mail/send
        response = sg.client.mail.send.post(request_body=mail_json)
        if response.status_code >= 200 and response.status_code <= 300:
            print("Send test email: Success!")
        else:
            print("Send test email: Failure ({response.status_code}, {response.text})")
    else:
        print("No sendgrid api key found. Skipping email")
