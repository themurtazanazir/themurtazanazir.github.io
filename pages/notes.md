---
layout: page
title: Notes
permalink: /notes/
---

# Notes

Welcome to the {{ site.title }} Notes pages! Here you can quickly jump to a 
particular page.

<div class="section-index">
    <hr class="panel-line">
    {% assign posts = site.notes | where: "coll","notes" %}
    {% for post in posts %}        
    <div class="entry">
    <h5><a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a></h5>
    <p>{{ post.description }}</p>
    </div>{% endfor %}
</div>
