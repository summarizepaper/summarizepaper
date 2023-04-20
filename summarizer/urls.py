from django.urls import path, re_path
from . import views
from django.contrib.auth import views as auth_views
from .views import RegisterView, ActivateView, CustomLoginView, logout_view
from django.views.generic import TemplateView


urlpatterns = [
    path('', views.summarize, name='summarize'),
    path('robots.txt/', views.robots_txt),
    #path('arxiv-id/(?P<arxiv_id>[\w\-/]+v\d+)/', views.arxividpage, name="arxividpage"),
    path("arxiv-id/<str:arxiv_id>/", views.arxividpage, name="arxividpage"),
    #path("arxiv-id/<str:cat>/<str:arxiv_id>/", views.arxividpage, name="arxividpage"),
    re_path(r"^arxiv-id/(?P<cat>[^\d]+)/(?P<arxiv_id>[a-zA-Z0-9]+)/$", views.arxividpage, name="arxividpage"),
    path("arxiv-id/<str:arxiv_id>/<str:error_message>/", views.arxividpage, name="arxividpage"),
    path("tree/<str:arxiv_id>/", views.tree, name="tree"),
    #path("summary-pdf/<str:arxiv_id>/", views.arxividpage, name="arxividpage"),
    #path('arxiv-id/(?P<arxiv_id>[\w\-/]+v\d+)/', views.arxividpage, name="arxividpage"),
    path("search-results/", views.search_results, name="search_results"),
    path("create-embed/", views.create_embed, name="create_embed"),
    path("about/", views.about, name="about"),
    path("chat/", views.chat, name="chat"),
    path("faq/", views.faq, name="faq"),
    path("contact/", views.contact, name="contact"),
    path("history/", views.history, name="history"),
    path("privacy/", views.privacy, name="privacy"),
    path("legal-notice/", views.legal, name="legal"),
    path("update-cache/", views.update_cache, name="update_cache"),
    path("vote/<str:paper_id>/", views.vote, name="vote"),
    #path('login/', auth_views.LoginView.as_view(), name='login'),
    path('login/', CustomLoginView.as_view(), name='login'),
    path('logout/', logout_view, name='logout'),
    path('register/', RegisterView.as_view(), name='register'),
    path('password_reset/', auth_views.PasswordResetView.as_view(template_name='password_reset_form.html'), name='password_reset'),
    path('password_reset/done/', auth_views.PasswordResetDoneView.as_view(template_name='password_reset_done.html'), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(template_name='password_reset_confirm.html'), name='password_reset_confirm'),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(template_name='password_reset_complete.html'), name='password_reset_complete'),
    path('activate/<uidb64>/<token>/', ActivateView.as_view(), name='activate'),
    path('account_activation_sent/', TemplateView.as_view(template_name='account_activation_sent.html'), name='account_activation_sent'),
    path('account_activation_invalid/', TemplateView.as_view(template_name='account_activation_invalid.html'), name='account_activation_invalid'),
]
