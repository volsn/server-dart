from django.urls import path
from main.views import IndexView, ArticlesView, output_view, ArticleView

urlpatterns = [
    path('', IndexView.as_view(), name='index'),
    path('articles/', ArticlesView.as_view(), name='articles'),
    path('article/<article>/', ArticleView.as_view(), name='article'),
    path('output/', output_view, name='output'),
]
