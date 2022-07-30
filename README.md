
<h1> Projeto de Iniciação Científica </h1>
This repository contain 2 projects about Reinforcement Learning, more specifically of Q-Learning, which I did with my mentor Lucas Pedroso.

<h1> Cartpole </h1>

<p> This project is about a fairly common algorithm to understand DQN - Double Q-Networks, where the objective of the game <br>
is that the cart and pole doesn't move a lot. If the pole fall beyond a certain angle it's game over and the samme applies <br>
when the cart moves too far away</p>

<p> In the project I basically defined the fundamental things about Q-Learning and later on just called Neural Networs to train the model</p>

<h1> Freezing Lake</h1>


<p> This code is to show the basics of Q-Learning, especially Q-Table, without using any complicated means.</p>

<p> This game is basically one where you threw a frisbie to your dog in the middle of a ice lake and now you need to go get it,<br>
your dog is a lazy one. You can move to 9 places and the actions are left, right, up and down. <br>
There's 3 types of blocks, the blocks in which you can stand on, those that has a hole and it's game over if you go on them and <br>
the last one where the frisbie is.
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

<b>Ambiente</b> &rarr; É o conjunto de todas as regras e ações (jogadas) que você tem no xadrez <br> 
<b>Agente </b>  &rarr; É quem interage com o ambiente e toma ações (você ou um algoritmo). <br> 
<b>Estado</b>   &rarr; É onde o agente se encontra para tomar ações. Por exemplo, quando o jogo de xadrez começa quem joga primeiro são quem está do lado das peças pretas. Logo, o estado em que o agente se encontra, a pessoa que vai jogar, é o estado inicial e ela pode fazer as ações permitidas. <br> 
<b>Ação</b> &rarr; É o ato de jogar. No nosso caso, é mover uma peça, seja um peão, cavalo. <br> 
<b>Recompensa</b> &rarr; É a forma de condicionar o agente a chegar em um certo objetivo. <br> 
Imagina que você tem um pet e você quer que ele te dê a patinha toda vez que você der a mão para ele. Uma das formas de fazer isso, é que toda vez que ele da a patinha, você da um petisco para ele, ou seja, você esta condicionando para um certo objetivo que é dar a patinha toda vez que você estender a mão para ele. <br>
 
A recompensa é exatamente isso, é o petisco que nós decidimos para o algoritmo para que ele alcance algum objetivo. <br>
Por exemplo, ganhar no jogo de xadrez ou dirigir de uma forma segura em um self driving car.
</p>


<h2> Objetivo </h2>

Mas qual o objetivo do agente? Uma resposta razoável seria imaginar que ele tentaria maximizar a sua recompensa, afinal essa é a métrica que estamos usando para ele chegar no nosso objetivo. <br> 

De fato, vamos fazer isso, só que com uma leve mudança, vamos maximizar a recompensa <b>cumulativa. </b> <br>

O motivo disso é que a recompensa é um caminho para um objetivo, é um processo que no fim chegamos no lugar desejado. Logo, não queremos maximizar só a recompensa <b>imediata</b>, um petisquinho que demos para o pet, mas sim a recompensa <b>cumulativa</b> até o fim, todos os petisquinhos até que o pet dê a patinha toda vez.<br>

Podemos definir esse conceito de uma maneira mais formal e temos o <b> retorno esperado </b>. <br>

Note que o nosso T, o número de recompensas possíveis, é finito. Porém, imagine um jogo de xadrez ou um cúbo de Rubick, apesar das possibilidades serem finitas elas são gigantescas. Tão grandes que nenhum computador atual consegue computar todas as possibilidades. <br>

Então, apesar do nosso número de recompensa, T, ser finito, ele acaba sendo infinito na prática.<br>

Como resolvemos isso? <br>

A ideia é bem simples, basta adicionar um termo, &gamma, que reduz os valores conforme T vai crescendo. Com isso, a partir de um certo ponto o T não vai somar nada significativo a nossa soma e vai sumir. <br>

Com isso em mente, temos agora o <b> retorno esperado descontado </b>. <br>

Podemos ver matematicamente que quando o nosso &gamma é &lt; 1, apesar da soma ser infinita, o retorno sera finito, pois essa soma converge.

<h2> Voltando as MDP'S </h2>
 
<p> Outro questionamento que podemos fazer agora é pensar em como o agente vai tomar uma decisão. Qual critério ele usa para agir? e como o algortimo calcula esse critério? <br>
 
 Para isso vamos introduzir um conceito super importante: <u> Política. </u> <br>
 
 <h2> Política </h2>
 
 A política vem responder a seguinte perguntas <br> 
 
 <b align="center"> Qual a probablidade de um agente tomar alguma ação em algum estado?</b> <br>
 
 Agora sim, temos um métrica para o nosso agente. Porém, como o algoritmo calcula essa probabilidade? <br>
 
 Para isso, vamos entender melhor a política. <br>
 
 A política é como se fosse um mapa, um guia para o nosso agente. É através dela que o agente toma as decisões. <br>
 
 Formalmente, podemos dizer que se o agente segue a política <b>&pi;</b> no tempo <b> t </b>, então <b>&pi;(a|s)</b> é a probabilidade de <b>A<sub>t</sub></b> = <b>a</b> se <b>S<sub>t</sub></b> = <b>s</b>. Isso significa que, no tempo <b>t</b>, sobre a política <b>&pi;</b>, a probabilidade de tomar a ação <b>a</b> no estado <b>s</b> é <b>&pi;(a|s)</b>. <br>
 
 Olhando de uma forma estatísca, para cada <b>s</b> &isin; <b>S</b>, <b>&pi;</b> é uma probabilidade de distribuição sobre <b>a</b> &isin; <b>A(s)</b>. <br><br>
 
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
A função valor estado para política a <b>&pi;</b>, denotada como $\large {\color{BurntOrange} v_{\pi}(s)}$, nos diz quão bom é um certo estado <b>s</b> para o nosso agente, quando o mesmo segue a política <b>&pi;</b>. Ou seja, nos retorna um valor de um estado sobre a política <b>&pi;.</b> <br>

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

Agora que temos as nossas métricas, um mapa que contém as probabilidades de quão bom é uma certa ação dado um certo estado para algum número de estados possíveis que podemos jogar, podemos pensar em como otimizar essa métrica. <br>
 
 </p>
 
 <h2> Otimização </h2>
 <p>
 
 Algo natural de se fazer agora que definimos as métricas pela qual o nosso agente vai tomar decisões, é tentar achar a melhor fórmula, ou seja, achar as políticas e funções valores ótimas tal que o meu agente tem a maior recompensa possível.
 
 </p>
<h3> <b>Política ótima</b></h3>
 <p>
De uma forma natural, podemos definir a política ótima $\large{\color{RubineRed}\pi}$ olhando o retorno esperado. Ou seja, $\large {\color{RubineRed}\pi}$ $\large\ge$ $\large{\color{Goldenrod} \pi'}$ se o <b>retorno esperado</b> de $\large {\color{RubineRed}\pi}$ é maior que o <b> retorno esperado</b> de $\large{\color{Goldenrod} \pi'}$  para todos os estados $s \in S$. Matematicamente: <br><br>
 
 $$\large {\color{RubineRed}\pi} \ge {\color{Goldenrod} \pi'} \ \text{se e somente se} \ {\color{RubineRed} v_{\pi}(s)} \ge {\color{Goldenrod} v_{\pi'}(s)} \ \text{para todo} \ s \in S$$
 
   Lembre-se que a nossa função $\large{\color{BurntOrange} v_{\pi}}$ volta o retorno esperado quando começamos no estado <b>s</b> e seguimos a política <b>&pi;</b>.<br> Simplesmente chamamos de política ótima aquela que é maior ou igual a todas as outras políticas.
    </p>
 <h3>$\large {\color{OrangeRed}\text{Função valor-estado ótima}}$</h3>
 <p>
 Vamos pegar o máximo de todas as nossas funções valor-ação, <b>v<sub>&pi;</sub>(s)</b>, e esse será a nossa função ótima:<br><br>
 
 $$\Large {\color{OrangeRed}v_{\ast}(s)} = \max_{\pi}\ {\color{BurntOrange} v_{\pi}(s)}$$
 
  Isso ocorre para todo <b>s</b> &in; S. Em outras palavras, <b>v<sub>*</sub></b> gera o maior retorno esperado possível para qualquer ação <b>s</b> sobre qualquer política <b>&pi;</b>.
 
 </p>
  
 <h3>$\large {\color{Goldenrod} \text{Função valor-ação ótima}}$</h3> 
 
 
 <p>
  De forma análoga ao que fizemos, vamos definir a nossa função valor-ação ótima como simplesmente o máximo das funções valor-ação. <br><br>
 
  $$ \Large {\color{Goldenrod} q_{\ast}(s,a)} = \max_{\pi}\ {\color{ForestGreen} q_{\pi}(s,a) }$$
  
  para todo <b>s</b> &in; em <b>S</b> e todo <b>a</b> &in; <b>A</b>. Em outras palavras, $\large {\color{Goldenrod} q_{\ast}}$ gera o maior retorno esperado possível para qualquer par ação-estado <b>(s,a)</b> sobre qualquer política <b>&pi;</b>.<br><br>
 
 Como vimos antes, a função $\large{\color{ForestGreen} q_{\pi}}$ nos dá o maior retorno esperado para qualquer par de ação <b>(s,a)</b> seguindo uma política <b>&pi;</b> qualquer.<br><br>
 
 A função $\large {\color{Goldenrod} q_{\ast}}$ gera o maior retorno esperado para qualquer par de estado-ação, <b>(s,a)</b>, olhando para todas as políticas <b>&pi;</b>. <br><br>
 
 Uma propriedade sensacional que $\large {\color{Goldenrod} q_{\ast}}$ possui é que ela deve satisfazer a equação ótima de <b>Bellman</b>. <br><br>
 
 $$ \Large  {\color{Goldenrod} q_{\ast}(s,a)} = E \left[ {\color{YellowGreen}R_{t+1} } + {\color{CornflowerBlue}\gamma \max_{a'}\ q_{\ast}(s',a') }\right]$$
 
 A equação de Bellman nos diz que para qualquer par de estado-ação, <b>(s,a)</b>, no tempo <b>t</b> o  <b>retorno esperado</b>, $\large {\color{Goldenrod} q_{\ast}}$ será:
 
 <ul>
 <li>$\large {\color{YellowGreen}R_{t+1}}$ &rarr; Recompensa de tomar a ação <b>a</b> no estado <b>s</b>. </li>
  <li>$\large {\color{CornflowerBlue}T}$ &rarr; O máximo do retorno esperado descontado para qualquer par de estado-ação <b>(s',a')</b>. </li>
 </ul>
  
Com $\large {\color{CornflowerBlue} T} = {\color{CornflowerBlue}\gamma \max_{a'}\ q_{\ast}(s',a') } $.
 
 Note que como estamos seguindo a política ótima, o proximo estado <b>s'</b> vai ser o estado ótimo onde podemos escolher a melhor ação <b>a'</b> em <b>t + 1</b>. <br><br>
 
</p> 
 
 <h1> Q-Tables </h1>
 <p>
 
Imagine que o computador está jogando um jogo onde ele pode abrir 3 portas e as ações que ele pode tomar são  $$\Large {\color{white} \uparrow \ \downarrow \ \rightarrow \ \leftarrow }$$

 <div align="center">
 <img align="center"  height = 400 width = 400 src="https://github.com/MateusBalotin/IC/blob/main/images/tables2.jpg" title="Jogo de abrir 3 portas"> 
 </div>
 <br>

 Note que podemos ver esses quadrados da tabela como estados. <br>
 
 <div align="center">
 <img align="center"  height = 400 width = 400 src="https://github.com/MateusBalotin/IC/blob/main/images/estadostable.jpg" title="Estados da tabela"> 
 </div>
 
 <br>
 
 <div align="center">
 <img align="center"  height = 400 width = 400 src="https://github.com/MateusBalotin/IC/blob/main/images/estaadoscomportas.jpg" title="Estados com as portas"> 
 </div>
 <br>
 
 
O objetivo do jogo é que o agente acumule pontos.

 <ul>
  <li> 30 pontos o agente ganha o jogo. </li>
  <li> -30 pontos ele perde o jogo.   </li>
 </ul>
 
Inicialmente o agente é como um bebê que acabou de nascer, ele não sabe o objetivo do jogo, qual é a melhor ação ou quais as recompensas cada estado tem. Porém, conforme ele vai <b> explorando </b> o ambiente, ele vai descobrindo tudo isso! <br>

Dessa forma ele consegue achar uma rota onde ele maximiza sua recompensa.<br>

O nosso jogo pode ser representado por essa Q-Table, onde temos todos os estados e ações possíveis. <br>
 
 <table class="content-table">
  <thead>
    <tr>
     <th> </th>
      <th>Cima</th>
      <th>Baixo</th>
      <th>Direta</th>
      <th>Esquerda</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Estado 1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
 <tr>
      <td>Estado 2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
 <tr>
      <td>Porta 1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
 <tr>
      <td>Estado 4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
 <tr>
      <td>Estado 5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
 <tr>
      <td>Estado 6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
 <tr>
      <td>Porta 2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
 <tr>
      <td>Estado 8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
 <tr>
      <td>Porta 3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>


Vamos supor que a primeira ação do agente é abrir a porta do meio. <br>

<div align="center">
 <img align="center"  height = 300 width = 400 src="https://github.com/MateusBalotin/IC/blob/main/images/room_minus10.jpg">
 </div>
<br>

Como podemos observar, ele recebeu <b>-10 pontos</b> de recompensa. Logo, faz sentido ele escolher outras portas além dessas. <br>

Agora ele escolheu a porta 1. <br>

<div align="center">
 <img align="center"  height = 300 width = 400 src="https://github.com/MateusBalotin/IC/blob/main/images/room10.jpg">
 </div>
 <br>
 
Muito bom!! Ele recebeu uma recompensa de <b>+10 pontos</b>, então ele só precisa continuar escolhendo a porta 1 para ganhar. Mas será que essa é a maneira mais eficiente de ele jogar? <br>

Vamos dar uma bisbilhotada no que a porta 3 está escondendo.<br>

<div align="center">
 <img align="center"  height = 300 width = 400 src="https://github.com/MateusBalotin/IC/blob/main/images/room30.jpg">
 </div>
<br>

Olha só! Atrás da porta 3 tem com uma recompensa incrível de <b>+30 pontos</b>. Se o agente tivesse escolhido a porta 3 desde o começo ele teria alcançando seu objetivo com 1 ação. <br>

Como podemos ver, seria bom que o nosso agente explorasse o ambiente, já que como vimos outros caminhos podem ter recompensas melhores. Caso ele não explorasse o ambiente e so escolhesse a porta 1, ele nunca teria achado a recompensa da porta 3.<br>

 
 </p>
 
 <h1> Exploration vs Exploitation </h1>
<p>
 
 
 
 </p>
 



