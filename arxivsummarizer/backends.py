from django.contrib.auth.backends import ModelBackend
from django.contrib.auth import get_user_model
from django.core.exceptions import PermissionDenied

User = get_user_model()

class CustomModelBackend(ModelBackend):
    def authenticate(self, request, username=None, password=None, **kwargs):
        print('auth')
        try:
            user = User.objects.get(username=username)
            if user.check_password(password):
                #if not user.is_active:
                #    print('raise')
                #    raise PermissionDenied("Your account is not activated yet.")
                return user
            else:
                return None
        except User.DoesNotExist:
            print('except')
            return None
