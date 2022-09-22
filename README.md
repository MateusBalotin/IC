
<h1> Projeto de Iniciação Científica </h1>
<p> 
 
Esse repositório é um relatório da minha Iniciação Científica com o professor Lucas G. Pedroso com o intuito de prover uma introdução ao Aprendizado por Reforço.
 
Nesse projeto é apresentado o código para o treinamento de 2 jogos: Cartpole e Freezing Lake.<br>
 </p>

<h1> CartPole </h1>

Cartpole é um jogo presente no enviroment gym, no qual é possível controlar somente 2 teclas, esquerda e direita.O jogo é composto de um carrino e uma vara em cima do mesmo, a qual pende o lado contrário do movimento.  <br>

O objetivo do jogo é não deixar a vara passar de um certo grau e não sair muito longe do centro da tela. <br>

Para o treinamento do mesmo, foi usado uma técnica bem comum, a DQN - Double Q-Network. A qual é basicamente uma junção dos conceitos de Aprendizado com Reforço com Redes Neurais. <br>

<h1> Freezing Lake</h1>


<p> O propósito desse código é monstrar os conceitos básicos do Q-Learning e um exemplo de uma Q-Table.</p>

<p> A história do jogo é que você foi brincar com o seu cachorro e arremesou o frisbie no meio de um lago congelado e o seu cachorro não está muito afim de ir pegar, então a sua missão é a recuperação do mesmo.<br>
O jogo é composto de 16 blocos, onde temos 3 tipos de blocos. O primeiro é o qual você pode andar, o segundo são os buracos que se você pisar, o episódio acaba, e o último onde o frisbie se encontra. <br>
</p>


<h1 align="center"> Q-Learning</h1>

<p> O aprendizado por reforço é uma área do Machine Learning que estuda como um agente em um ambiente, consegue maximizar uma certa recompensa. <br>

As aplicações do aprendizado por reforço são variadas, desde de jogos como Mario, Dota, até jogos de tabuleiro como Go, Xadrez e self-driving cars, como os da Tesla.
</p>

<h1 align="center"> Markov Decision Process – MDP’S </h1>

<p> Um dos componentes principais do Aprendizado por Reforço são os processos de decisão de Markov:</p>

<ul> 
<li> Ambiente</li>
<li> Agente</li>
<li> Estado</li>
<li> Ação</li>
<li> Recompensa</li>
</ul>

<p>Vamos entender isso com um exemplo. Imagine que você está jogando xadrez com um amigo. <br>

<b>Ambiente</b> &rarr; É o conjunto de todas as regras e ações (jogadas) que você tem no xadrez. <br> 
<b>Agente </b>  &rarr; É quem interage com o ambiente e toma ações, você ou um algoritmo. <br> 
<b>Estados</b>   &rarr; São os lugares possíveis em que o agente pode estar. Por exemplo, no jogo de xadrez temos um tabuleiro com 64 casas, onde casa representa um estado. Em outros jogos, cada pixel pode representar um estado diferente. <br> 
<b>Ações </b> &rarr; É o ato de jogar, por exemplo mexer um peão ou um cavalo. <br> 
<b>Recompensa</b> &rarr; É a forma de condicionar o agente a chegar em um certo objetivo. Imagine que você tem um pet e você quer que ele te dê a patinha toda vez que você der a mão para ele. Uma das formas de fazer isso, é que toda vez que ele da a patinha, você da um petisco para ele, ou seja, você esta condicionando ele para um certo objetivo que é dar a patinha toda vez que você estender a mão para ele. <br>
 
A recompensa é exatamente isso, é o petisco que nós decidimos para o algoritmo para que ele alcance algum objetivo. <br>
 
</p>
<h2> Objetivo </h2>

Mas qual o objetivo do agente? Uma resposta razoável seria imaginar que ele tentaria maximizar a sua recompensa, afinal essa é a métrica que estamos usando para ele chegar no nosso objetivo. <br> 

De fato, vamos fazer isso, só que com uma leve mudança, vamos maximizar a <b>recompensa cumulativa. </b> <br>

O motivo disso é que a recompensa é um caminho para um objetivo, é um processo que no fim chegamos no lugar desejado. Logo, não queremos maximizar só a recompensa <b>imediata</b>, um petisquinho que demos para o pet, mas sim a recompensa <b>cumulativa</b> até o fim, todos os petisquinhos até que o pet dê a patinha toda vez.<br>

Podemos definir esse conceito de uma maneira mais formal e temos o <b> retorno esperado. </b> <br>

De uma forma mais formal, definimos o <b> recompensa esperada </b> como <br>

$$ {\color{Violet}G_{t}} = R + R_{1} + R_{2} + ... + R_{t}$$

para um tempo t, onde $R_{i}, i = 1, 2, ..., t$ é a recompensa ganha em cada t. <br>

Note que o nosso T, o número de recompensas possíveis, é finito. Porém, imagine um jogo de xadrez ou um cúbo de Rubick, apesar das possibilidades serem finitas elas são gigantescas. Tão grandes que nenhum computador atual consegue computar todas as possibilidades. <br>

Então, apesar do nosso número de recompensa, T, ser finito, ele acaba sendo infinito na prática.<br>

Como resolvemos isso? <br>

A ideia é bem simples, basta adicionar um termo, &\gamma$, que reduz os valores conforme T vai crescendo. Com isso, a partir de um certo ponto o T não vai somar nada significativo a nossa soma e vai sumir. <br>

Com isso em mente, temos agora o <b> retorno esperado descontado </b>, que pode ser definido formalmente da seguinte forma <br>

$$ {\color{Violet}G_{t}}  = R + \gamma \cdot R_{1} + \gamma^{2} \cdot R_{2} + ... $$
$$ {\color{Violet}G_{t}} = \sum_{k=0}^\infty \gamma^{k} R_{t+k+1} } $$

Podemos ver matematicamente que quando o nosso &gamma é &lt; 1$, apesar da soma ser infinita, o retorno sera finito, pois essa soma converge.

<h2> Voltando as MDP'S </h2>
 
<p> Outro questionamento que podemos fazer agora é pensar em como o agente vai tomar uma decisão. Qual critério ele usa para agir? e como o algoritmo calcula esse critério? <br>
 
 Para isso vamos introduzir um conceito super importante: <u> Política. </u> <br>
 
 <h2> Política </h2>
 
 A política vem responder a seguinte perguntas <br> 
 
 <b align="center"> Qual a probablidade de um agente tomar alguma ação em algum estado?</b> <br>
 
 Agora sim, temos um métrica para o nosso agente. Porém, como o algoritmo calcula essa probabilidade? <br>
 
 Para isso, vamos entender melhor a política. <br>
 
 A política é como se fosse um mapa, um guia para o nosso agente. É através dela que o agente toma as decisões. <br>
 
 Formalmente, podemos dizer que se o agente segue a política <b>&pi;</b> no tempo <b> t </b>, então <b>&pi;(a|s)</b> é a probabilidade de <b>A<sub>t</sub></b> = <b>a</b> se <b>S<sub>t</sub></b> = <b>s</b>. Isso significa que, no tempo <b>t</b>, sobre a política <b>&pi;</b>, a probabilidade de tomar a ação <b>a</b> no estado <b>s</b> é <b>&pi;(a|s)</b>. <br>
 
 Olhando de uma forma estatísca, para cada <b>s</b> &isin; <b>S</b>, <b>&pi;</b> é uma probabilidade de distribuição sobre <b>a</b> &isin; <b>A(s)</b>. <br>
 
 Para isso, vamos inserir um conceito que da mais sentindo a política, <b> funções-valores. </b> <br>
 
 </p>
 <h2> Função valor-ação e valor-estado </h2>
 
 <p>
 
 As funções valores vem responder a seguinte pergunta: <br>
 
 <b align="center"> Quão bom é uma certa ação ou estado para o agente.</b> <br>
 
Com essa respostas podemos dar sentido a política. Afinal, se sabemos quão bom é uma certa ação ou estado para o agente, podemos decidir sempre ir pelo caminho que <b>maximizamos</b> a nossa recompensa. No final, é exatamente isso que vamos fazer!! <br>
 
Vamos explorar como são definidas essas funções valores!
 
 
</p>
<h3> $\large{\color{BurntOrange} \text{Função valor-estado}}$</h3>
 
 <p>
A função valor estado para política a <b>&pi;</b>, denotada como $\large {\color{BurntOrange} v_{\pi}(s)}$, nos diz quão bom é um certo estado <b>s</b> para o nosso agente, quando o mesmo segue a política <b>&pi;</b>. Ou seja, nos retorna um valor de um estado sobre a política <b>&\pi$.</b> <br>

Formalmente, o valor de um estado sobre a política <b>&pi;</b> é o <b>retorno esperado</b> de quando começamos no estado <b>s</b> no tempo <b>t</b> e depois seguimos a política <b>&pi;</b>. Matematicamente definimos $\large {\color{BurntOrange} v_{\pi}(s)}$ como: <br>

  $$ \Large {\color{BurntOrange} v_{\pi}(s)} = E_{\pi} \left [ {\color{Violet} G_{t} }| S_{t} = s \right] $$ 
 
  $$ \Large {\color{BurntOrange} v_{\pi}(s) }= E_{\pi} \left [ {\color{Violet} \sum_{k=0}^\infty \gamma^{k} R_{t+k+1} }| S_{t} = s \right ] $$


De forma similiar podemos definir a Função valor-ação.
</p>



<h3> $\large{\color{ForestGreen} \text{Função valor-ação}}$</h3>
<p> 

 A Função valor-ação para a política <b>&pi;</b>, denotada como <b>q<sub>&pi;</sub></b>, nos diz quão bom é para o agente tomar uma certa ação em um certo estado seguindo a política <b>&pi;</b>. Analogamente ao caso anterior, isso nos retorna o valor de uma ação sobre a política <b>&pi</b>;.<br>
 
 Formalmente, podemos definir o valor de uma ação <b>a</b> em um estado <b>s</b> sobre a política <b>&pi;</b> como o <b> retorno esperado </b> de quando começamos no estado <b>s</b> no tempo <b>t</b>, tomando uma ação <b>a</b> e seguindo a política <b>&pi;</b>. Matematicamente podemos definir <b>q<sub>&pi;</sub>(s,a)</b> como: <br>
 
  $$\Large {\color{ForestGreen} q_{\pi}(s,a) } = E_{\pi} \left [ {\color{Violet} G_{t} } | S_{t} = s, A_{t} = a \right]  $$ 
 
  $$\Large {\color{ForestGreen} q_{\pi}(s,a) } = E_{\pi} \left [ {\color{Violet} \sum_{k=0}^\infty \gamma^{k} R_{t+k+1} }| S_{t} = s, A_{t} = a \right ] $$
 
Temos nomes especiais para essa $\large {\color{ForestGreen}\text{função}}$ e o $\large{\color{Violet}\text{retorno esperado}}$:
 
 <ul>
  <li> $\large {\color{ForestGreen}q_{\pi}(s,a)}$  &rarr; $\normalsize \textbf{Q-Função}$</li>
  <li> $\large {\color{Violet} G_{t}}$&rarr; $\normalsize \textbf{Q-Valor}$</li>
</ul>
 
 
Opa! Encontramos algo interessante nessa nomenclatura. Uma <b>Q-Função</b> com um <b>Q-Valor</b> e o nosso método é chamado <b>Q-Learning</b>. Não pode ser coincidência né? <br>

Esses <b>Q-Valores</b> são exatamente quem vamos buscar para ter o melhor guia para o nosso agente! <br>

Agora que temos as nossas métricas, um mapa que contém as probabilidades de quão bom é uma certa ação dado um certo estado, podemos pensar em como otimizar essa métrica. <br>
 
 </p>
 
 <h2> Otimização </h2>
 <p>
 
 Algo natural após definirmos as métricas pela qual o nosso agente vai tomar decisões, é tentar achar a melhor fórmula, ou seja, achar as políticas e funções valores ótimas tal que o agente tem a maior recompensa possível.
 
 </p>
<h3> <b>Política ótima</b></h3>
 <p>
De uma forma natural, podemos definir a política ótima $\large \pi$ olhando o retorno esperado. Ou seja, $\large \pi$ $\large\ge$ $\large \pi'$ se o <b>retorno esperado</b> de $\large \pi$ é maior que o <b> retorno esperado</b> de $\large \pi'$  para todos os estados $s \in S$. Matematicamente <br><br>
 
 $$\large \pi \ge \pi' \ \text{se e somente se} \  v_{\pi}(s) \ge  v_{\pi'}(s) \ \text{para todo} \ s \in S.$$
 
   Lembre-se que a nossa função $\large v_{\pi}$ volta o retorno esperado quando começamos no estado <b>s</b> e seguimos a política <b>&pi;</b>.<br> Simplesmente chamamos de política ótima aquela que é maior ou igual a todas as outras políticas.
 </p>
 <h3>$\large {\color{OrangeRed}\text{Função valor-estado ótima}}$</h3>
 <p>
 Vamos pegar o máximo de todas as nossas funções valor-ação, <b>v<sub>&pi;</sub>(s)</b>, e esse será a nossa função ótima:<br><br>
 
 $$\Large {\color{OrangeRed}v_{\ast}(s)} = \max_{\pi}\ {\color{BurntOrange} v_{\pi}(s)}$$
 
  Isso ocorre para todo <b>s</b> &in; S. Em outras palavras, $v_{*}$ gera o maior retorno esperado possível para qualquer ação <b>s</b> sobre qualquer política <b>&pi;</b>.
 
 </p>
  
 <h3>$\large {\color{Goldenrod} \text{Função valor-ação ótima}}$</h3> 
 
 
 <p>
  De forma análoga ao que fizemos, vamos definir a nossa função valor-ação ótima como simplesmente o máximo das funções valor-ação. <br><br>
 
  $$\Large {\color{Goldenrod} q_{\ast}(s,a)} = \max_{\pi}\ {\color{ForestGreen} q_{\pi}(s,a)}$$
  
  para todo <b>s</b> &in; em <b>S</b> e todo <b>a</b> &in; <b>A</b>. Em outras palavras, $\large {\color{Goldenrod} q_{\ast}}$ gera o maior retorno esperado possível para qualquer par ação-estado <b>(s,a)</b> sobre qualquer política <b>&pi;</b>.<br>
 
 Como vimos antes, a função $\large{\color{ForestGreen} q_{\pi}}$ nos dá o maior retorno esperado para qualquer par de ação <b>(s,a)</b> seguindo uma política <b>&pi;</b> qualquer.<br>
 
 A função $\large {\color{Goldenrod} q_{\ast}}$ gera o maior retorno esperado para qualquer par de estado-ação, <b>(s,a)</b>, olhando para todas as políticas <b>&pi;</b>. <br>
 
 Uma propriedade sensacional que $\large {\color{Goldenrod} q_{\ast}}$ possui é que ela deve satisfazer a equação ótima de <b>Bellman</b>, <br>
 
 $$\Large  {\color{Goldenrod} q_{\ast}(s,a)} = E \left[ {\color{YellowGreen}R_{t+1} } + {\color{CornflowerBlue}\gamma \max_{a'}\ q_{\ast}(s',a') }\right].$$
 
 A equação de Bellman nos diz que para qualquer par de estado-ação, <b>(s,a)</b>, no tempo <b>t</b> o  <b>retorno esperado</b>, $\large {\color{Goldenrod} q_{\ast}}$ será:
 
 <ul>
 <li>$\large {\color{YellowGreen}R_{t+1}}$ &rarr; Recompensa de tomar a ação <b>a</b> no estado <b>s</b>. </li>
  <li>$\large {\color{CornflowerBlue}T}$ &rarr; O máximo do retorno esperado descontado para qualquer par de estado-ação <b>(s',a')</b>. </li>
 </ul>
  
Com $\large {\color{CornflowerBlue} T} = {\color{CornflowerBlue}\gamma \max_{a'}\ q_{\ast}(s',a')}$.
 
 Note que como estamos seguindo a política ótima, o proximo estado <b>s'</b> vai ser o estado ótimo onde podemos escolher a melhor ação <b>a'</b> em <b>t + 1</b>. <br><br>
 
</p> 
 
 <h1> Q-Learning </h1>
 <p>
 
 Com tudo isso que vimos, temos que uma forma de achar a política ótima é achar os melhores Q-Valores para cada par de estado-ação, ou seja, o maior retorno esperado descontado para qualquer par de estado-ação. <br>
 
 Com os Q-Valores em mãos, basta atualizá-los iterativamente até que convergam para a equação de Bellman!! <br>
 
 Mas como podemos interprar isso matematicamente e entender o conceito mais claramente? <br>
 
Para isso imagine que o computador está jogando um jogo onde os estados possíveis são quadrados em brancos e 3 portas. As ações que ele pode tomar são  $$\Large {\color{white} \uparrow \ \downarrow \ \rightarrow \ \leftarrow }$$

 O jogo é bem simples, o jogador vai começar em algum quadrado e se movimentará com as ações acima. <br>
 
 <div align="center">
 <img align="center"  height = 400 width = 400 src="https://github.com/MateusBalotin/IC/blob/main/images/tables2.jpg" title="Jogo de abrir 3 portas"> 
 </div>
 <br>

 Note que podemos ver esses quadrados da tabela como estados. <br>
 
 <div align="center">
 <img align="center"  height = 400 width = 400 src="https://github.com/MateusBalotin/IC/blob/main/images/estadostable.jpg" title="Estados da tabela"> 
 </div>
 <br>
 Colocando nossas portas nessa tabela, temos: <br><br>
 
 <div align="center">
 <img align="center"  height = 400 width = 400 src="https://github.com/MateusBalotin/IC/blob/main/images/estaadoscomportas.jpg" title="Estados com as portas"> 
 </div>
 <br>
 
 
O jogo funciona assim: <br>

<ul>
<li>$\large \textrm{Objetivo}$ $\rightarrow$ Somar 30 pontos</li>
</ul>
<br>

| Recompensas | Pontuação |
| --- | --- |
| $$\textrm{Estados}$$ | $${\color{red}-1}$$ |
| $${\color{yellow} \textrm{Porta 1}}$$ | $${\color{Green}-10}$$ |
| $${\color{red}\textrm{Porta 2}}$$ | $${\color{Green}+10} $$|
| $${\color{pink}\textrm{Porta 3}}$$ | $${\color{Red}+30}$$|

<br>

$\large \textrm{Regras}$
<ol>
 <li> A movimentação só poder ser feita normalmente, ou seja, se estiver no Estado 1 e fizer a ação  $\Large {\color{white} \uparrow}$ você não saí do lugar, as únicas ações possíveis para se movimentar são  $\Large {\color{white} \downarrow \ \rightarrow }$.</li>
 <li> É possível ficar entrando e saindo de uma porta. Por exemplo, no Estado 2 ao tomar a ação $\Large {\color{white} \rightarrow }$  você abre a ${\color{GreenYellow} \textrm{Porta} \ 1}$. Para abrí-la novamente basta voltar uma casa e avançar, isto é, tomar as ações $\Large {\color{white}  \leftarrow \ \rightarrow}$ ou $\Large {\color{white} \downarrow \ \uparrow }$.</li>
 </ol>
 
Inicialmente o agente é como um bebê que acabou de nascer, ele não sabe o objetivo do jogo, qual é a melhor ação ou quais as recompensas cada estado tem. Porém, conforme ele vai <b> explorando </b> o ambiente, ele vai descobrindo tudo isso! <br>

Logo, conforme ele vai jogando partidas, ele vai descobrindo os Q-Valores para cada par de estado-ação e assim ele consegue achar a melhor rota que maximiza sua recompensa. <br>

Para fins de organização e visualização, vamos guardar nossos Q-Valores nessa Q-Tabela, onde no eixo horizontal estão os estados e no eixo horizontal as possíveis ações.<br>


 <div align="center">
 <img align="center"  height = 400 width = 400 src="https://github.com/MateusBalotin/IC/blob/main/images/Primordial_Genesis.png"> 
 </div>
 <br>

Pronto! Agora só falta o computador começar a jogar e conforme ele for jogando, ele:

<ol> 
  <li>Vai aprender novos Q-Valores para cada par de estado-ação. </li>
  <li>Atualizará esses Q-Valores de forma iterativa. </li>
 <li> Em cada rodada vai escolher a ação com maior Q-Valor</li>
 </ol>
 
 Mas pera aí! Como que vai funcionar a primeira jogada? Afinal, no começo o computador não sabe nada sobre o jogo e todos os Q-Valores estão zerados. Como ele vai escolher o melhor Q-Valor então? <br>
 
 <div align="center">
 <img align="center"  height = 300 width = 400 src="https://github.com/MateusBalotin/IC/blob/main/images/confused-meme.gif"> 
 </div>
 <br>

 Para responder essa pergunta vamos definir o conceito de ${\color{Fuchsia}\textrm{Exploration}}$ vs ${\color{Emerald}\textrm{Exploitation}}$.
 
 </p>
 
 <h2> ${\color{Fuchsia}\textrm{Exploration}}$ vs  ${\color{Emerald}\textrm{Exploitation}}$</h2>
<p>
 
 Como você pode imaginar, ${\color{Fuchsia}\textrm{Exploration}}$, é o ato de explorar o ambiente, ou seja, não vamos nos preocupar com os Q-Valores ao tomarmos uma ação. <br>
 
 Porém, ${\color{Emerald}\textrm{Exploitation}}$, é exatemente o contrário. É o ato de utilizarmos o que sabemos do ambiente para a nossa vantagem, escolher as ações baseados na recompensa máxima que teremos delas. Ou seja, escolher as ações que tem o maior Q-Valor para cada par de estado-ação. <br>

Logo, no começo do jogo, o computador irá explorar o ambiente, não se preocupando com os Q-Valores. Assim, com cada jogada ele vai adquirindo novos Q-Valores para os pares de estado-ação e descobrindo as recompensas para os diferentes estados do jogo. <br>
 
Ufa! Agora faz sentido como o computador faz para jogar na primeira rodada. Mas como podemos definir essa taxa de ${\color{Fuchsia}\textrm{Exploration}}$ vs ${\color{Emerald}\textrm{Exploitation}}$ no código? <br>
 
 Para isso, vamos implementar uma Greedy Epsilon Strategy, ou no bom português, uma Estatégia Gananciosa Epsilon. <br>
 
 Apesar do nome um pouco assustador, a implementação é bem simples. <br>
 
 <ol>
 <li> Defina Epsilon $(\large \epsilon)$ = 1 </li>
 <li> Escolha um número aleatório r entre 0 e 1</li>
 <li> Se $\large \epsilon$ > r $\rightarrow$ Então vamos explorar o ambiente</li>
  <li> Diminua o valor de $\large \epsilon$ a cada episódio</li>
  </ol>
 
 Assim, no começo do jogo exploramos mais o ambiente, descobrindo os Q-Valores para cada estado e conforme vamos jogando, esses Q-Valores vão sendo atualizados iterativamente até que convergam para o seu valor ótimo. <br>
 
 Quando temos esses Q-Valores em mãos, como a taxa de exploração vai decaindo, vamos tomar nossas decisões baseadas nesses Q-Valores!! <br>
 </p>
 
 <h1> Voltando ao nosso jogo das Portas</h1>
 <p>
 
 Finalmente, vamos ver como seria a primeira partida. Vamos dizer que o computador tomou a ação ${\color{white} \downarrow \ },$ ou seja, foi do Estado 1 para o Estado 4. <br>
 
 Como atualizamos o Q-Valor para essa jogada? <br>
 
 Vamos nos recordar da equação de Bellman: <br>
 
  $$ \Large  {\color{Goldenrod} q_{\ast}(s,a)} = E \left[ {\color{YellowGreen}R_{t+1} } + {\color{CornflowerBlue}\gamma \max_{a'}\ q_{\ast}(s',a') }\right]$$
 
 Como vimos, a equação de Bellman nos conta a seguinte história: <br>
 
 <ul>
 <li>$\large {\color{YellowGreen}R_{t+1}}$ &rarr; Recompensa de tomar a ação <b>a</b> no estado <b>s</b>. </li>
  <li>$\large {\color{CornflowerBlue}T}$ &rarr; O máximo do retorno esperado descontado para qualquer par de estado-ação <b>(s',a')</b>. </li>
 </ul>
 
 Queremos que os nossos Q-Valores se aproximem do lado direito da equação de Bellman para que assim convergam para o nosso Q-Valor ótimo. <br>
 
 Só falta um detalhe para a receita ficar pronta!
 
 <h3> ${\color[rgb]{0.71, 0.4, 0.11}\textrm{Taxa de Aprendizado}}$ </h3>
 
 Imagine que temos um Q-Valor para um certo par de estado-ação. <br>
 
 Ao voltarmos nesse estado em outro episódio, teremos um novo Q-Valor para esse mesmo par de estado-ação. Como prosseguimos? Ignoramos completamente o valor antigo ou o consideramos nos novos cálculos? <br>
 
 A ${\color[rgb]{0.71, 0.4, 0.11}\textrm{taxa de aprendizado}}$, Learning Rate, vai ser a medida que nos dirá o quão rápido vamos abandonar o nosso Q-Valor atual para o próximo. <br>
 
 Ela será denotada pela letra grega $\Large \alpha$. <br>
 
 Para o nosso episódio vamos considerar $\large \alpha = 0.8$.
 
 Agora sim, estamos com todos os ingredientes e podemos começar a nossa receita!!<br>
 
 A fórmula que utilizaremos para calcular o novo Q-Valor do par estado-ação (s,a) no tempo t é: <br>
 
 <!-- $$\large{\color{Goldenrod} q^{new}(s,a)} = (1 - \alpha) \underbrace{q(s,a)}_{\textrm{valor antigo}} + \overbrace{\left({\color{YellowGreen}R\_{t+1} } + {\color{CornflowerBlue}\gamma \max\_{a'}\ q(s',a')}\right)}^{\textrm{valor novo}}$$ -->
 
   $$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}(s,a)} = (1 - \alpha) \underbrace{q(s,a)}_{\color[rgb]{0.47, 0.41, 0.47}\textrm{valor antigo}} + \alpha \overbrace{\left(R\_{t+1} + \gamma \max\_{a'}\ q(s',a')\right)}^{\color[rgb]{0.0, 0.4, 0.65}\textrm{valor novo}}$$

Essa fórmula é bem parecida com a equação de Bellman, só foi acrescentado a ${\color[rgb]{0.71, 0.4, 0.11}\textrm{taxa de aprendizado}}$ e o nosso ${\color[rgb]{0.47, 0.41, 0.47}\textrm{Q-Valor antigo}}$.<br>

Note que quando temos $\large \alpha = 1$ a primeira parte do lado direito da equação, o ${\color[rgb]{0.47, 0.41, 0.47}\textrm{valor antigo}}$ se anula e só temos o ${\color[rgb]{0.0, 0.4, 0.65}\textrm{Q-Valor novo}}$. <br>

Da mesma forma, $\large \alpha = 0$ significa que nunca vamos atualizar o nosso Q-Valor.<br>

Mas chega de conversa! Vamos calcular o Q-Valor para quando o computador começa no Estado 1 e toma a ação ${\color{white} \downarrow}$. Vamos supor que $\gamma = 0.99$. <br>

$$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}(s,a)} = (1 - \alpha) \underbrace{q(s,a)}_{\color[rgb]{0.47, 0.41, 0.47}\textrm{valor antigo}} + \alpha \overbrace{\left(R\_{t+1} + \gamma \max\_{a'}\ q(s',a')\right)}^{\color[rgb]{0.0, 0.4, 0.65}\textrm{valor novo}}$$

$$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}(s,a)} = (1 - 0.8) (0) + 0.8 \left\[-1 + 0.99 \left( \max\_{a'}\ q(s',a')\right)\right]$$

Quem exatamente é $\max\_{a'}\ q(s',a')$?<br>

Note que como todos os Q-Valores são 0, temos:

$$\large \max\_{a'}\ q(s',a') = \max (q\ (\textrm{estado4}, {\color{white} \uparrow}), \ q(\textrm{estado4},{\color{white} \downarrow} ), \ q(\textrm{estado4}, {\color{white} \rightarrow}), \ q(\textrm{estado4},{\color{white} \leftarrow})) $$


$$\large \max\_{a'}\ q(s',a') = \max (0, 0, 0, 0) $$

$$\large \max\_{a'}\ q(s',a') = 0 $$

Agora com o valor de $\max\_{a'}\ q(s',a')$ em mãos, podemos achar o nosso Q-Valor. <br>

$$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}(s,a)} = (1 - 0.8) (0) + 0.8 \left\[-1 + (0.99 \cdot 0) \right]$$

$$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}(s,a)} = (0.2)(0) + 0.8 \cdot -(1) $$

$$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}(s,a)} = -0.8 $$

Logo, o nosso Q-Valor para o par estado-ação $\large (s,a)$ = $(\textrm{Estado 1}, {\color{white} \downarrow})$ = -0.8. <br>

Agora vamos repetir esse processo até o jogo acabar ou podemos definir um limite de passos máximo que o jogador pode ter. <br>

Observe que quando os nossos Q-Valores convergirem para os Q-Valores ótimos, ou seja, a nossa Q-Função convergir para a Q-Função ótima, teremos a nossa tão sonhada política ótima. <br>

A nossa Q-Tabela que guarda os nossos Q-Valores está assim no momento: <br>


 <div align="center">
 <img align="center"  height = 300 width = 400 src="https://github.com/MateusBalotin/IC/blob/main/images/Primordial.jpg"> 
 </div>
 <br>
 
 O computador está no Estado 4 agora e toma a mesma ação ${\color{white} \downarrow}$. <br>
 
 Logo, como vimos o Q-Valor para essa par de estado-ação é: 
 
 $$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}(s,a)} = (1 - \alpha) \underbrace{q(s,a)}_{\color[rgb]{0.47, 0.41, 0.47}\textrm{valor antigo}} + \alpha \overbrace{\left(R\_{t+1} + \gamma \max\_{a'}\ q(s',a')\right)}^{\color[rgb]{0.0, 0.4, 0.65}\textrm{valor novo}}$$

$$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}(\textrm{Estado 4},\downarrow)} = (1 - 0.8) (0) + 0.8 \left\[10 + 0.99 \left( \max\_{a'}\ q({\color{red}\textrm{Porta 2}},a')\right)\right]$$

Note que $\max\_{a'}\ q(s,a') = \max\_{a'}\ q(\textrm{Estado 6},\textrm{a'})$ vai ser 0 de forma análoga ao caso passado. <br>

Logo o Q-Valor para o nosso par de estado-ação $(\textrm{Estado 4}, \downarrow)$ é:

$$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}(\textrm{Estado 4},\downarrow)} = (0.8)(10)$$

$$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}(\textrm{Estado 4},\downarrow)} = 8$$

Agora que o computador recebeu uma recompensa positiva da ${\color{red} \textrm{Porta 2}}$, vamos supor que ele irá voltar para o Estado 4. <br>

Logo o Q-Valor de $({\color{red}\textrm{Porta 2}}, \uparrow)$ é:

 $$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}(s,a)} = (1 - \alpha) \underbrace{q(s,a)}_{\color[rgb]{0.47, 0.41, 0.47}\textrm{valor antigo}} + \alpha \overbrace{\left(R\_{t+1} + \gamma \max\_{a'}\ q(s',a')\right)}^{\color[rgb]{0.0, 0.4, 0.65}\textrm{valor novo}}$$

$$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}({\color{red}\textrm{Porta 2}}\uparrow)} = (1 - 0.8)(0) + 0.8 \left\[-1 + 0.99 \left( \max\_{a'}\ q(\textrm{Estado 4},a')\right)\right]$$

Note que $\max\_{a'}\ q(\textrm{Estado 4},a')$ será:

$$\large \max\_{a'}\ q(\textrm{Estado 4},a') = \max (q\ (\textrm{estado4}, {\color{white} \uparrow}), \ q(\textrm{estado4},{\color{white} \downarrow} ), \ q(\textrm{estado4}, {\color{white} \rightarrow}), \ q(\textrm{estado4},{\color{white} \leftarrow})) $$

$$\large \max\_{a'}\ q(\textrm{Estado 4},a') = \max (0, 8, 0, 0) $$

$$\large \max\_{a'}\ q(\textrm{Estado 4},a') = 8 $$

Note que agora temos um valor para o par estado-ação $(\textrm{Estado 4}, \downarrow).$<br>

Logo, o nosso Q-Valor para $({\color{red}\textrm{Porta 2}}, \uparrow)$ é:

$$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}({\color{red}\textrm{Porta 2}}, \uparrow)} = (0.2)(0) + (0.8)(-1) + 8 $$

$$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}({\color{red}\textrm{Porta 2}}, \uparrow)} = -0.8  + 8$$

$$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}({\color{red}\textrm{Porta 2}}, \uparrow)} = 7.2$$

Vamos dar um olhada em como esta nossa Q-Tabela: <br>

 <div align="center">
 <img align="center"  height = 400 width = 400 src="https://github.com/MateusBalotin/IC/blob/main/images/tabela%202%20vdd.jpg" title="Estados com as portas"> 
 </div>
 <br>
 
 Como o computador recebeu uma recompensa alta abrindo a ${\color{red} \textrm{Porta 2}}$ ele irá ficar voltando nesse estado até ganhar. <br>
 
 A sua recompensa atual é: $R_{Estado 4} + R_{{\color{red}\textrm{Porta 2}}} = -1 + 10 = 9$. <br>
 
Já que o computador quer voltar a ${\color{red} \textrm{Porta 2}}$ e ele se encontra no Estado 4, ele irá tomar a ação ${\color{white} \downarrow}$.

$$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}(s,a)} = (1 - \alpha) \underbrace{q(s,a)}_{\color[rgb]{0.47, 0.41, 0.47}\textrm{valor antigo}} + \alpha \overbrace{\left(R\_{t+1} + \gamma \max\_{a'}\ q(s',a')\right)}^{\color[rgb]{0.0, 0.4, 0.65}\textrm{valor novo}}$$

$$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}(\textrm{Estado 4}\downarrow)} = (1 - 0.8)(8) + 0.8 \left\[10 + 0.99 \left( \max\_{a'}\ q({\color{red}\textrm{Porta 2}},a')\right)\right]$$

Vamos calcular quem é $\max\_{a'}\ q({\color{red}\textrm{Porta 2}},a')$. <br>

$$\large \max\_{a'}\ q({\color{red}\textrm{Porta 2}},a') = \max (q\ ({\color{red}\textrm{Porta 2}}, {\color{white} \uparrow}), \ q({\color{red}\textrm{Porta 2}},{\color{white} \downarrow} ), \ q({\color{red}\textrm{Porta 2}}, {\color{white} \rightarrow}), \ q({\color{red}\textrm{Porta 2}},{\color{white} \leftarrow})) $$

$$\large \max\_{a'}\ q({\color{red}\textrm{Porta 2}},a') = \max (7.2, 0, 0, 0) $$

$$\large \max\_{a'}\ q({\color{red}\textrm{Porta 2}},a') = 7.2 $$

Logo, o novo Q-Valor para o par de estado-ação (Estado 4, ${\color{white} \downarrow})$ é:

$$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}(\textrm{Estado 4}, \uparrow)} = (0.2)(8) + (0.8)(10) + 7.2$$

$$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}(\textrm{Estado 4}, \uparrow)} = 1.6  + 8 + 7.2$$

$$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}(\textrm{Estado 4}, \uparrow)} = 16.8 $$

Como discutimos, o computador irá ficar indo da ${\color{red}\textrm{Porta 2}}$ para o Estado 4. <br>

Logo o novo Q-Valor de $({\color{red}\textrm{Porta 2}}, \uparrow)$ é:

 $$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}(s,a)} = (1 - \alpha) \underbrace{q(s,a)}_{\color[rgb]{0.47, 0.41, 0.47}\textrm{valor antigo}} + \alpha \overbrace{\left(R\_{t+1} + \gamma \max\_{a'}\ q(s',a')\right)}^{\color[rgb]{0.0, 0.4, 0.65}\textrm{valor novo}}$$

$$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}({\color{red}\textrm{Porta 2}}\uparrow)} = (1 - 0.8)(7.2) + 0.8 \left\[-1 + 0.99 \left( \max\_{a'}\ q(\textrm{Estado 4},a')\right)\right]$$

Note que $\max\_{a'}\ q(\textrm{Estado 4},a')$ será:

$$\large \max\_{a'}\ q(\textrm{Estado 4},a') = \max (q\ (\textrm{estado4}, {\color{white} \uparrow}), \ q(\textrm{estado4},{\color{white} \downarrow} ), \ q(\textrm{estado4}, {\color{white} \rightarrow}), \ q(\textrm{estado4},{\color{white} \leftarrow})) $$

$$\large \max\_{a'}\ q(\textrm{Estado 4},a') = \max (0, 16.8, 0, 0) $$

$$\large \max\_{a'}\ q(\textrm{Estado 4},a') = 16.8 $$

Note que agora temos um valor para o par estado-ação $(\textrm{Estado 4}, \downarrow)$.<br>

Logo, o nosso Q-Valor para $({\color{red}\textrm{Porta 2}}, \uparrow)$ é:

$$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}({\color{red}\textrm{Porta 2}}, \uparrow)} = (0.2)(7.2) + (0.8)(-1) + 16.8 $$

$$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}({\color{red}\textrm{Porta 2}}, \uparrow)} = 1.4 - 0.8 + 16.8 $$

$$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}({\color{red}\textrm{Porta 2}}, \uparrow)} = 17.4$$

A recompensa atual é 27.<br>

Finalmente, de forma análoga anterior temos que o novo Q-Valor para $(\textrm{Estado 4}, \downarrow)$. <br>

$$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}(\textrm{Estado 4}\downarrow)} = 28.8$$

Como a nossa recompensa é mair que 30, o primeiro episódio acabou. <br>

A nossa Q-Tabela ficou assim: <br>

<div align="center">
 <img align="center"  height = 400 width = 400 src="https://github.com/MateusBalotin/IC/blob/main/images/A%C3%A7%C3%B5es-2.png"> 
</div>

<br>

Vamos repetir esse proceso para mais 2 partidas, porém com ações diferentes. <br>

Imagine que agora o computador queira chegar na $\textrm{\color{yellow}Porta 1}$. Para isso, precisamos calcular o valor de $Q(\textrm{Estado 1}, \rightarrow)$. <br>

Como os cálculos são extensivos e são análogos aos já feitos, serão omitidos. <br>

$$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}(\textrm{Estado 1}\rightarrow)} = -0.8$$

$$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}(\textrm{Estado 2}\rightarrow)} = 8$$

$$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}(\textrm{{\color{yellow}Porta 2}}\leftarrow)} = 7.2$$

De forma similiar ao anterior, o computador ficará voltando para a Porta 2 até o episódio acabar. <br>

$$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}(\textrm{Estado 2}\rightarrow)} = 16.8$$

$$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}(\textrm{{\color{yellow}Porta 2}}\leftarrow)} = 17.4$$

$$\large{\color[rgb]{1.0, 1.0, 0.2} q^{new}(\textrm{Estado 2}\rightarrow)} = 28.8$$

Note que como a recompensa é maior que 30 o episódio acabou. Como as portas tem a mesma recompensa e a distância até elas é a mesma, os resultados foram iguais. <br>

Por fim, no último episódio o computador vai ir até a $\textrm{\color{pink}Porta 3}$. <br>

Para chegar até la, suas ações serão: $\rightarrow \ \downarrow \ \downarrow \ \rightarrow$.

Logo, temos os seguintes Q-Valores. <br>

 <ol>
 <li>$Q(\textrm{Estado 1}, \rightarrow) = 27.8$ </li>
 <li> $Q(\textrm{Estado 2}, \downarrow) \ \ = -0.8$</li>
 <li> $Q(\textrm{Estado 5}, \downarrow) \ \  = -0.8$</li>
 <li> $Q(\textrm{Estado 8}, \rightarrow) = 24$</li>
 <li> $Q(\textrm{{\color{pink}Porta 3}}, \leftarrow) \ \ =  23.2 $</li></li>
 <li> $Q(\textrm{Estado 8}, \rightarrow) = 51.8$</li>
  </ol>
  
 Como a nossa recompensa pasou de 30 o episódio terminou. A Q-Tabela está dessa forma:
  
 <div align="center">
 <img align="center"  height = 400 width = 400 src="https://github.com/MateusBalotin/IC/blob/main/images/a%C3%A7%C3%B5es-3.png"> 
</div>

 
 </p>



