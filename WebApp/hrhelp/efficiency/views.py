from django.shortcuts import render, render_to_response
from django.template.context_processors import csrf
from django.contrib import auth
from django.http import HttpResponseRedirect
from sklearn.externals import joblib


def index(request):
    return render(request, 'main/main.html')


def login(request):
    args = {}
    args.update(csrf(request))
    if request.method == "POST":
        username = request.POST.get("username", "")
        password = request.POST.get("password", "")
        user = auth.authenticate(username=username, password=password)
        if user is not None:
            auth.login(request, user)
            return HttpResponseRedirect("/")
        else:
            args['login_error'] = "Не верный логин или пароль"
            return render_to_response('login.html', args)

    else:
        return render(request, 'login.html')


def logout(request):
    auth.logout(request)
    return HttpResponseRedirect("/")


def efficiency(request):
    uvolnenie = None
    percent = None
    if request.method == 'POST':
        form_dict = {}
        form_dict['fio'] = request.POST.get("fio")
        form_dict['position'] = request.POST.get("position")
        form_dict['workyears'] = int(request.POST.get("workyears"))
        form_dict['ispolnitel'] = int(request.POST.get("ispolnitel"))
        form_dict['communication'] = int(request.POST.get("communication"))
        form_dict['stress'] = int(request.POST.get("stress"))
        form_dict['lider'] = int(request.POST.get("lider"))
        form_dict['poryadochnost'] = int(request.POST.get("poryadochnost"))
        form_dict['energy'] = int(request.POST.get("energy"))
        form_dict['syd'] = request.POST.get("syd")
        form_dict['vigovor'] = request.POST.get("vigovor")
        form_dict['zaslugi'] = request.POST.get("zaslugi")
        form_dict['dopobraz'] = request.POST.get("dopobraz")

        syd = form_dict['syd']
        if syd == None:
            syd = 0
        else:
            syd = 1
        vigovor = form_dict['vigovor']
        if vigovor == None:
            vigovor = 0
        else:
            vigovor = 1
        zaslugi = form_dict['zaslugi']
        if zaslugi == None:
            zaslugi = 0
        else:
            zaslugi = 1
        dopobraz = form_dict['dopobraz']
        if dopobraz == None:
            dopobraz = 0
        else:
            dopobraz = 1
        tree_model = joblib.load('efficiency/data/tree_model.sav')

        data = [[form_dict['workyears'], syd, vigovor, zaslugi, dopobraz, form_dict['ispolnitel'],
                 form_dict['communication'], form_dict['stress'], form_dict['lider'], form_dict['poryadochnost'],
                 form_dict['energy']]]
        uvolnenie = tree_model.predict(data)
        percent = tree_model.predict_proba(data)
        if uvolnenie == 1:
            percent = round(percent[0][1] * 100, 2)
        else:
            percent = round(percent[0][0] * 100, 2)
    return render(request, 'main/main.html', {'uvolnenie': uvolnenie, 'percent': percent, 'form_dict': form_dict})
