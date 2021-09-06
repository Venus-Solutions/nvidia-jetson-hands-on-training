import requests

url = 'https://notify-api.line.me/api/notify'
token = 'YOUR_TOKEN'
headers = {'Authorization':'Bearer ' + token}

def send_msg(msg):
    r = requests.post(url, headers=headers, data = {'message':msg})
    return r

def send_pic_with_msg(pic_path, msg):
    r = requests.post(url, headers=headers, files={'imageFile': open(pic_path,'rb')}, data = {'message':msg})
    return r

if __name__ == "__main__":
    res = send_msg('YOUR_MESSAGE')
    # res = send_pic_with_msg('YOUR_PICTURE_FILE', 'YOUR_MESSAGE')
    print(res.text)