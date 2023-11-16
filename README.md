# BeautifulPrompt generator
An extension for [webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) that lets you generate prompts. Using llm to generate prompts.

![](screenshot.png)

## Getting more models online
You can add models from huggingface to the selection of models in setting. The setting field
is `Hugginface model names for promptgen, separated by comma`, and its default value is
just:
```
alibaba-pai/pai-bloom-1b1-text2prompt-sd-v2,alibaba-pai/pai-bloom-1b1-text2prompt-sd
```

[beautifulprompt](https://blog.csdn.net/u012193416/article/details/134358448?spm=1001.2014.3001.5501)


Reloading UI is required to apply this setting.

## Getting more models offline
Put your models into the `models` directort inside the extension. For example, my model can be
made fully offline by placing fines from https://huggingface.co/alibaba-pai/pai-bloom-1b1-text2prompt-sd-v2 into
those directories:

```

 📁 webui root directory
 ┗━━ 📁 extensions
     ┗━━ 📁 sd_webui_beautifulprompt
         ┗━━ 📁 models                            
             ┗━━ 📁 pai-bloom-1b1-text2prompt-sd-v2              <----- any name can be used
                 ┣━━ 📄 config.json               <----- each model has its own set of required files;
                 ┣━━ 📄 generation_config.json   
                 ┣━━ 📄 pytorch_model.bin                       mine requires all those
                 ┣━━ 📄 special_token_map.json
                 ┣━━ 📄 tokenizer_config.json
                 ┗━━ 📄 tokenizer.json
```

Reloading UI is required to see new models.