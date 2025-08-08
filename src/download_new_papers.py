# encoding: utf-8
import os
import tqdm
from bs4 import BeautifulSoup as bs
import re
import urllib.request
import json
import datetime
import pytz


def _download_new_papers(field_abbr):
    NEW_SUB_URL = f'https://arxiv.org/list/{field_abbr}/new'  # https://arxiv.org/list/cs/new
    page = urllib.request.urlopen(NEW_SUB_URL)
    soup = bs(page, features="html.parser")
    content = soup.body.find("div", {'id': 'content'})

    # find the first h3 element in content
    h3 = content.find("h3").text   # e.g: New submissions for Wed, 10 May 23
    date = h3.replace("New submissions for", "").strip()

    dt_list = content.dl.find_all("dt")
    dd_list = content.dl.find_all("dd")
    arxiv_base = "https://arxiv.org/abs/"

    assert len(dt_list) == len(dd_list)
    new_paper_list = []
    for i in tqdm.tqdm(range(len(dt_list))):
        paper = {}
        # Robust id / URL extraction
        abs_href = None
        for a in dt_list[i].find_all('a', href=True):
            href = a['href']
            if '/abs/' in href:
                abs_href = href
                break
        paper_id = ""
        if abs_href:
            # Normalize href and id
            if abs_href.startswith('http'):
                # absolute URL
                paper_id = abs_href.rstrip('/').split('/')[-1]
                main_url = abs_href
            else:
                # relative URL like /abs/2508.01234
                paper_id = abs_href.rstrip('/').split('/')[-1]
                main_url = f"https://arxiv.org{abs_href}"
        else:
            # Fallback to regex on text
            m = re.search(r"abs/(\d{4}\.\d+|[a-z\-]+/\d+)", dt_list[i].get_text(" ", strip=True))
            if m:
                paper_id = m.group(1)
                main_url = f"https://arxiv.org/abs/{paper_id}"
            else:
                main_url = arxiv_base

        paper['main_page'] = main_url
        paper['pdf'] = f"https://arxiv.org/pdf/{paper_id}.pdf" if paper_id else "https://arxiv.org/pdf/"

        # Normalize helpers
        def norm_text(s: str) -> str:
            return re.sub(r"\s+", " ", s or "").strip()

        title_div = dd_list[i].find("div", {"class": "list-title mathjax"})
        title_raw = title_div.get_text(" ", strip=True) if title_div else ""
        title_clean = re.sub(r"^Title:\s*", "", norm_text(title_raw))
        paper['title'] = title_clean

        authors_div = dd_list[i].find("div", {"class": "list-authors"})
        authors_raw = authors_div.get_text(" ", strip=True) if authors_div else ""
        authors_clean = norm_text(re.sub(r"^Authors:\s*", "", authors_raw))
        paper['authors'] = authors_clean

        subjects_div = dd_list[i].find("div", {"class": "list-subjects"})
        subjects_raw = subjects_div.get_text(" ", strip=True) if subjects_div else ""
        subjects_clean = norm_text(re.sub(r"^Subjects:\s*", "", subjects_raw))
        paper['subjects'] = subjects_clean

        abs_p = dd_list[i].find("p", {"class": "mathjax"})
        abstract_raw = abs_p.get_text(" ", strip=True) if abs_p else ""
        paper['abstract'] = norm_text(abstract_raw)

        # Optional comments (often contain venue info, project/code links)
        comments_div = dd_list[i].find("div", {"class": "list-comments"})
        if comments_div and comments_div.text:
            comments_raw = comments_div.text
            comments_clean = norm_text(re.sub(r"^Comments:\s*", "", comments_raw))
            paper['comments'] = comments_clean
            # Extract first URL (project/code page) if any
            url_match = re.search(r"https?://[^\s>]+", comments_clean)
            paper['project_url'] = url_match.group(0) if url_match else ""
        else:
            paper['comments'] = ""
            paper['project_url'] = ""


        print(f"DEBUG: {paper['comments']}")

        new_paper_list.append(paper)


    #  check if ./data exist, if not, create it
    if not os.path.exists("./data"):
        os.makedirs("./data")

    # save new_paper_list to a jsonl file, with each line as the element of a dictionary (KST)
    date = datetime.date.fromtimestamp(datetime.datetime.now(tz=pytz.timezone("Asia/Seoul")).timestamp())
    date = date.strftime("%a, %d %b %y")
    with open(f"./data/{field_abbr}_{date}.jsonl", "w") as f:
        for paper in new_paper_list:
            f.write(json.dumps(paper) + "\n")


def get_papers(field_abbr, limit=None):
    # KST 기준 파일명
    date = datetime.date.fromtimestamp(datetime.datetime.now(tz=pytz.timezone("Asia/Seoul")).timestamp())
    date = date.strftime("%a, %d %b %y")
    if not os.path.exists(f"./data/{field_abbr}_{date}.jsonl"):
        _download_new_papers(field_abbr)
    results = []
    with open(f"./data/{field_abbr}_{date}.jsonl", "r") as f:
        for i, line in enumerate(f.readlines()):
            if limit and i == limit:
                return results
            results.append(json.loads(line))
    return results
