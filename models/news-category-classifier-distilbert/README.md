---
license: apache-2.0
metrics:
- accuracy
- f1
---
Returns news category given text.

See https://www.kaggle.com/code/dima806/news-category-classification-distilbert for more details.

```
Classification report:

                precision    recall  f1-score   support

          ARTS     0.4985    0.5629    0.5288       302
ARTS & CULTURE     0.5061    0.4627    0.4834       268
  BLACK VOICES     0.5903    0.4776    0.5280       917
      BUSINESS     0.6136    0.5659    0.5888      1198
       COLLEGE     0.5043    0.5066    0.5054       229
        COMEDY     0.5990    0.5630    0.5804      1080
         CRIME     0.6365    0.6615    0.6488       712
CULTURE & ARTS     0.7133    0.4744    0.5698       215
       DIVORCE     0.8498    0.8015    0.8249       685
     EDUCATION     0.5000    0.5025    0.5012       203
 ENTERTAINMENT     0.7383    0.8146    0.7745      3473
   ENVIRONMENT     0.5490    0.5433    0.5461       289
         FIFTY     0.6107    0.3250    0.4242       280
  FOOD & DRINK     0.7514    0.8320    0.7897      1268
     GOOD NEWS     0.4676    0.2321    0.3103       280
         GREEN     0.4685    0.5401    0.5018       524
HEALTHY LIVING     0.5669    0.4712    0.5147      1339
 HOME & LIVING     0.8267    0.8113    0.8189       864
        IMPACT     0.5000    0.3702    0.4254       697
 LATINO VOICES     0.6066    0.5664    0.5858       226
         MEDIA     0.6136    0.5688    0.5903       589
         MONEY     0.6193    0.5840    0.6012       351
     PARENTING     0.6711    0.7673    0.7160      1758
       PARENTS     0.5094    0.4475    0.4764       791
      POLITICS     0.8154    0.8365    0.8258      7120
  QUEER VOICES     0.7949    0.7392    0.7660      1269
      RELIGION     0.6681    0.6097    0.6376       515
       SCIENCE     0.6370    0.6327    0.6348       441
        SPORTS     0.7628    0.8079    0.7847      1015
         STYLE     0.6343    0.6231    0.6286       451
STYLE & BEAUTY     0.8656    0.8894    0.8773      1962
         TASTE     0.4701    0.4320    0.4502       419
          TECH     0.6188    0.5952    0.6068       420
 THE WORLDPOST     0.5786    0.5825    0.5806       733
        TRAVEL     0.8501    0.8596    0.8548      1980
     U.S. NEWS     0.4256    0.3018    0.3532       275
      WEDDINGS     0.8320    0.8810    0.8558       731
    WEIRD NEWS     0.5030    0.4559    0.4783       555
      WELLNESS     0.7272    0.8459    0.7821      3589
         WOMEN     0.4841    0.4062    0.4417       714
    WORLD NEWS     0.4936    0.4682    0.4806       660
     WORLDPOST     0.6840    0.6376    0.6600       516

      accuracy                         0.7073     41903
     macro avg     0.6275    0.5966    0.6080     41903
  weighted avg     0.7007    0.7073    0.7017     41903
```
