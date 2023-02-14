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
        help_text="Your password must be at least 8 characters and cannot be entirely numeric.",
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

    def clean_password1(self):
        password1 = self.cleaned_data.get("password1")
        if len(password1) < 8:
            raise forms.ValidationError("Password must be at least 8 characters.")

        if password1.isdigit():
            raise forms.ValidationError("Password cannot be entirely numeric.")

        if not any(char.isdigit() for char in password1):
            raise forms.ValidationError("Password must contain at least one number.")

        if not any(char.isupper() for char in password1):
            raise forms.ValidationError("Password must contain at least one uppercase letter.")

        if not any(char.islower() for char in password1):
            raise forms.ValidationError("Password must contain at least one lowercase letter.")

        if not any(char in "!@#$%^&*()" for char in password1):
            raise forms.ValidationError("Password must contain at least one special character.")

        return password1
