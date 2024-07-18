---
layout: page
permalink: /summaries/
title: Paper Summaries
description: >
nav: true
nav_order: 4
---

Summaries and critiques of papers relating to Machine Learning, usually Multimodal ML and Generative AI. I'm writing these summaries as a way of having an easy reference to past papers I've wrote and to (hopefully) expose people to some of the papers and ideas I find interesting. The summaries will be short, as they are inspired by the paper summaries done by <a href="https://fanpu.io/summaries/"> Fan Pu Zeng </a>, which are in turn inspired by <a href="https://www.cs.cmu.edu/~15712/summaries.html">this class at CMU</a>.

---

<ol>
    {% for summary in site.summaries reversed %}
    <li>
        <a href="{{ summary.url | relative_url }}">
            ({{ summary.date | date: '%b %-d, %Y' }})
            {{ summary.title }}
        </a>
    </li>
    {% endfor %}
</ol>
