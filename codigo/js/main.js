
function iniciar() {
    let botao = document.getElementById('submit');
    botao.addEventListener("click", () => { main() }
    )
}

function main() {

    let inputs = getInputs()
    let req = new Req(inputs)
    let response = requestNotas(req)
    inserirNotas(response)
}

function getInputs() {

    let values = []
    let inputs = document.getElementsByClassName('form_input')

    for (i = 0; i < inputs.length; i++) {

        values.push(inputs[i].value)
    }

    return values
}

class Req {

    constructor(dt) {
        this.TP_SEXO = parseInt(dt[0]);
        this.TP_COR_RACA = parseInt(dt[1]);
        this.TP_ESCOLA = parseInt(dt[2]);
        this.TP_LINGUA = parseInt(dt[3]);
        this.RENDA = parseInt(dt[4]);
        this.FOCO = parseInt(dt[5]);
        this.INFO = parseInt(dt[6]);
        this.CASA = parseInt(dt[7]);
        this.PAIS = parseInt(dt[8]);
    }
}



function requestNotas(info) {

    const url = new URL('https://henriqueangar.github.io/TERMOMETRO_ENEM/predict');
    for (const chave in info) {
        url.searchParams.append(chave, info[chave]);
    }

    console.log(url)
    fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Erro na solicitação: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            return data
        })
        .catch(error => {
            console.error('Erro na solicitação:', error);
        });
}


function inserirNotas(notas) {

    let areas = document.getElementsByClassName("nota_area-nota");

    areas[0].innerText = notas.ch
    areas[1].innerText = notas.cn
    areas[2].innerText = notas.mt
    areas[3].innerText = notas.lc
    areas[4].innerText = notas.red

    let soma = 0
    for (i = 0; i < areas.length; i++) {
        soma += parseInt(areas[i].innerText)
    }

    let media = Math.round(soma / 5);
    let geral = document.getElementById('geral')
    geral.innerText = media
}