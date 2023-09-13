
function iniciar() {
    let botao = document.getElementById('submit');
    botao.addEventListener("click", () => { calcularNota() }
    )
}

function calcularNota() {

    let values = []
    let inputs = document.getElementsByClassName('form_input')

    for (i = 0; i < inputs.length; i++) {

        values.push(inputs[i].value)
    }

    enviar(values)
}

function enviar(info) {
    console.log(info)
    inserirNotas()
}

function inserirNotas() {

    let soma = 0
    let areas = document.getElementsByClassName("nota_area-nota");

    for (i = 0; i < areas.length; i++) {
        areas[i].innerText = 500;
        soma += 500;
    }

    let media = soma / 5;
    let geral = document.getElementById('geral')
    geral.innerText = media
}