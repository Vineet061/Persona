import os
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()
foldName = os.environ.get("foldBB")

def docManager():
    try:
        docs = os.listdir(foldName)
        currentTime = datetime.now().strftime("%m-%d-%Y")
        formatString = "%m-%d-%Y"
        print(currentTime,"This is the currentTime")
        for img in docs:
            print(img,"This is the img name")
            imgName = img.split("_")[1]
            print(imgName)

            dt1 = datetime.strptime(imgName, formatString)
            dt2 = datetime.strptime(currentTime, formatString)

            daysDiff = (dt2-dt1).days
            print(daysDiff)
            if daysDiff>1:
                os.remove(os.path.join("imgBB",img))
        
    except Exception as e:
        return(f"Getting error {e}")










