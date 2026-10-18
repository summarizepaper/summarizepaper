from django import forms
from .models import Vote, User
from django.contrib.auth.forms import UserCreationForm

#class RegistrationForm(forms.Form):
#    username = forms.CharField(max_length=20)
#    email = forms.EmailField()
#    password1 = forms.CharField(widget=forms.PasswordInput)
#    password2 = forms.CharField(widget=forms.PasswordInput)

class RegistrationForm(UserCreationForm):
    username = forms.CharField(max_length=30, required=True)
    email = forms.EmailField(max_length=254, required=True)
    password1 = forms.CharField(
        widget=forms.PasswordInput,
        label="Password",
        strip=False,
        help_text="At least 8 characters, not a common password and not only digits.",
    )
    password2 = forms.CharField(
        widget=forms.PasswordInput,
        label="Password confirmation",
        strip=False,
        help_text="Enter the same password as before, for verification.",
    )

    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')
