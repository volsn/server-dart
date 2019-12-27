from django.urls import path
from main.views import index_view, articles_view, output_view, article_view

urlpatterns = [
    path('', index_view, name='index'),
    path('articles/', articles_view, name='articles'),
    path('article/<article>/', article_view, name='article'),
    path('output/', output_view, name='output'),
]
