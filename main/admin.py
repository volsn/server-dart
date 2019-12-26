from django.contrib import admin
from .models import Article

# Register your models here.

class ArticleAdmin(admin.ModelAdmin):
    list_filter = ('type',)

admin.site.register(Article, ArticleAdmin)
