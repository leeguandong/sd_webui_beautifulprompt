
function beautifulprompt_send_to(where, text){
    textarea = gradioApp().querySelector('#beautifulprompt_selected_text textarea')
    textarea.value = text
    updateInput(textarea)

    gradioApp().querySelector('#beautifulprompt_send_to_'+where).click()

    where == 'txt2img' ? switch_to_txt2img() : switch_to_img2img()
}

function beautifulprompt_send_to_txt2img(text){ beautifulprompt_send_to('txt2img', text) }
function beautifulprompt_send_to_img2img(text){ beautifulprompt_send_to('img2img', text) }

function submit_beautifulprompt(){
    var id = randomId()
    requestProgress(id, gradioApp().getElementById('beautifulprompt_results_column'), null, function(){})

    var res = create_submit_args(arguments)
    res[0] = id
    return res
}
