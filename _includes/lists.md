<!-- <div class="section-index"> -->
<!-- <hr class="panel-line"> -->
{% assign posts = site.notes | where: "coll",page.coll_name %}
{% for post in posts %}
<!-- <div class="entry"> -->
## [{{ post.title }}]({{ post.url | prepend: site.baseurl }})

{{ post.description }}. . .  [read more]({{ post.url | prepend: site.baseurl }})

<!-- </div> -->
{% endfor %}
<!-- </div> -->