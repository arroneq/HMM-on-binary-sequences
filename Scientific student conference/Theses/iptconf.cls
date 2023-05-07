%%% --------------------------------------------------------
%%> \section{Declaration}
%%% --------------------------------------------------------

\ProvidesExplClass{iptconf}{}{0.0.1}{IPT conference class}
\LoadClass[twocolumn, oneside]{article}


%%% --------------------------------------------------------
%%> \section{Class Options}
%%% --------------------------------------------------------
\RequirePackage{l3keys2e}

\DeclareOption{incollection}{%
	\pagestyle{empty}
}

\DeclareOption{single}{%
    \pagestyle{plain}
}

\ExecuteOptions{incollection}
\ProcessOptions\relax

%%% --------------------------------------------------------
%%> \section{Geometry Of Page}
%%% --------------------------------------------------------

\RequirePackage[%
	%showframe,
	a4paper,%
	footskip=1cm,%
	headsep=0.3cm,%
	top=2cm, %поле сверху
	bottom=2cm, %поле снизу
	left=2cm, %поле ліворуч
	right=2cm, %поле праворуч
]{geometry}
\let\@afterindentfalse\@afterindenttrue

%%% --------------------------------------------------------
%%> \section{Language Package}
%%% --------------------------------------------------------

\sys_if_engine_luatex:TF
{
	\RequirePackage{fontspec}
	\setsansfont{Open Sans}%{Arial}
	\setmainfont{CMU Serif}%{Times New Roman}
	\setmonofont{CMU Typewriter Text}%{Consolas}
	\defaultfontfeatures{Ligatures={TeX}}
	\RequirePackage[]{unicode-math}
	\setmathfont{XITS Math}
	\RequirePackage{lua-widow-control}
	\RequirePackage[protrusion=true, expansion, tracking=true]{microtype}
	\RequirePackage{amsmath}
}
{
	\RequirePackage[utf8]{inputenc}
	\RequirePackage[T2A,T1]{fontenc}
	\RequirePackage{cmap}
	\renewcommand{\sfdefault}{opensans-LF}
	\renewcommand{\bfdefault}{b}
	\RequirePackage[protrusion=true, expansion, tracking=true, kerning=true]{microtype}
	\RequirePackage{amsmath, amssymb}
}
\RequirePackage[english, russian, ukrainian]{babel}

%% В назві статті математичні символи також будуть жирними
\g@addto@macro\bfseries{\boldmath}


%%% --------------------------------------------------------
%%> \section{Language Dependencies}
%%% --------------------------------------------------------

\NewDocumentCommand{\newlanguagecommand}{mm}
{% #1 = command to define
	% #2 = key-value list
	\prop_new:c { \__iptconf_nlc_strip:N #1 }
	\prop_gset_from_keyval:cn { \__iptconf_nlc_strip:N #1 } { #2 }
	\NewDocumentCommand{#1}{o}
	{
		\IfNoValueTF { ##1 }
		{
			\iptconf_nlc:NV #1 \languagename
		}
		{
			\iptconf_nlc:Nn #1 { ##1 }
		}
	}
}

\cs_new:Nn \iptconf_nlc:Nn
{
	\prop_if_in:cnTF { \__iptconf_nlc_strip:N #1 } { #2 }
	{
		\prop_item:cn { \__iptconf_nlc_strip:N #1 } { #2 }
	}
	{
		UNSET ~ TRANSLATION
	}
}
\cs_generate_variant:Nn \iptconf_nlc:Nn { NV }

% syntactic sugar
\cs_new:Nn \__iptconf_nlc_strip:N { g_iptconf_nlc_ \cs_to_str:N #1 _prop }


%%% --------------------------------------------------------
%%> \subsection{LDefinition Language Dependence Macro}
%%% --------------------------------------------------------

\newlanguagecommand{ \iptbibname }
{
	ukrainian = Перелік ~ використаних ~ джерел,
	russian   = Перечень ~ использованных ~ источников,
	english   = References
}

\newlanguagecommand{ \keyname }
{
	ukrainian = Ключові ~ слова,
	russian   = Ключевые ~ слова,
	english   = Keywords
}

\newlanguagecommand{ \pacsnum }
{
	ukrainian = PACS ~ номери,
	russian   = PACS ~ номера,
	english   = PACS ~ numbers
}

\newlanguagecommand{ \udcnum }
{
	ukrainian = УДК,
	russian   = УДК,
	english   = UDC
}

\newlanguagecommand{ \ipt }
{
	ukrainian = Навчально-науковий ~ Фізико-технічний ~ інститут,
	russian   = Учебно-научный ~ Физико-технический ~ институт,
	english   = Institute ~ of ~ Physics ~ and ~ Technology
}


\NewDocumentCommand{ \PaperLanguage }{ m }{
	\selectlanguage{#1}
	\maketitle
}


%%% --------------------------------------------------------
%%> \section{Speaker Regform}
%%% --------------------------------------------------------

\iow_new:N   \g_iptconf_regform_iow
\iow_open:Nn \g_iptconf_regform_iow {\c_sys_jobname_str.txt}

\keys_define:nn { iptconf }
{
	fullname       .str_set:N = \l_iptconf_fullname_str,
	birthday       .str_set:N = \l_iptconf_birthday_str,
	position       .str_set:N = \l_iptconf_position_str,
	phone          .str_set:N = \l_iptconf_phone_str,
	authoremail    .str_set:N = \l_iptconf_authoremail_str,
	confsection    .str_set:N = \l_iptconf_confsection_str,
	copynum        .str_set:N = \l_iptconf_copynum_str,
	needliving     .str_set:N = \l_iptconf_needliving_str,
	needinvitanion .str_set:N = \l_iptconf_needinvitanion_str,
}

\DeclareDocumentCommand \regform { m }
{
	\keys_set:nn { iptconf } { #1 }
	\iow_now:Nx \g_iptconf_regform_iow
	{
		\text_expand:n {Повне ~ ім'я ~ доповідача: ~    \str_use:N \l_iptconf_fullname_str}       \iow_newline:
		\text_expand:n {Дата ~ народження: ~            \str_use:N \l_iptconf_birthday_str}       \iow_newline:
		\text_expand:n {Посада ~ доповідача: ~          \str_use:N \l_iptconf_position_str}       \iow_newline:
		\text_expand:n {Телефонний ~ номер: ~           \str_use:N \l_iptconf_phone_str}          \iow_newline:
		\text_expand:n {Email: ~                        \str_use:N \l_iptconf_authoremail_str}    \iow_newline:
		\text_expand:n {Секція: ~                       \str_use:N \l_iptconf_confsection_str}    \iow_newline:
		\text_expand:n {Кількість ~ копій ~ збірника: ~ \str_use:N \l_iptconf_copynum_str}        \iow_newline:
		\text_expand:n {Потреба ~ в ~ житлі: ~          \str_use:N \l_iptconf_needliving_str}     \iow_newline:
		\text_expand:n {Потреба ~ в ~ запрошенні: ~     \str_use:N \l_iptconf_needinvitanion_str}
	}
	\iow_close:N \g_iptconf_regform_iow
}


%%% --------------------------------------------------------
%%> \section{Bibliography}
%%% --------------------------------------------------------

\RequirePackage[backend=biber, %
	bibstyle=gost-numeric,
	maxnames = 10,
	babel=other,
	isbn=true,
	url=true,
	doi=true,
%	defernumbers=true,
    sorting=none,
	backend=biber,
]{biblatex}


%%% --------------------------------------------------------
%%> \subsection{Bibliography Typesetting}
%%% --------------------------------------------------------

\setlength\bibitemsep{0.5\itemsep}
\setlength{\biblabelsep}{1ex}
\AtEndDocument{
	\printbibliography[title = \iptbibname]
}


%%% --------------------------------------------------------
%%> \section{Lists}
%%% --------------------------------------------------------

\RequirePackage[]{enumitem}
\setlist{nosep}


%%% --------------------------------------------------------
%%> \section{Colums}
%%% --------------------------------------------------------

\RequirePackage{flushend}


%%% --------------------------------------------------------
%%> \section{Making Title}
%%% --------------------------------------------------------

%%% --------------------------------------------------------
%%> \subsection{Variables}
%%% --------------------------------------------------------

\tl_new:N    \l_udc_tl
\tl_new:N    \l_pacs_tl
\tl_new:N    \l_abstract_tl
\tl_new:N    \l_keywords_tl
\tl_new:N    \l_author_tl
\tl_new:N    \l_author_email_tl
\tl_new:N    \l_author_affiliation_tl
\tl_new:N    \l_title_tl
\tl_new:N    \l_title_clean_tl
\tl_new:N    \l_affiliation_tl
\tl_new:N    \l_affiliation_number_tl
\seq_new:N   \g_authors_seq
\seq_new:N   \g_affiliation_seq

\iow_new:N   \g_totoc_iow
\iow_open:Nn \g_totoc_iow {\c_sys_jobname_str.dat}

\AtEndDocument{
    \iow_now:Nx \g_totoc_iow { \iow_char:N \\finishauthors }
    \iow_close:N \g_totoc_iow
}

%%% --------------------------------------------------------
%%> \subsection{Grab macros}
%%% --------------------------------------------------------

\RenewDocumentCommand{\title}{ m }
{
	\tl_set:Nn \l_title_tl {#1}
    \tl_set:Nn \l_title_clean_tl {#1}
    \regex_replace_all:nnN { \ * \c{\\} \ * } { \ } \l_title_clean_tl
	\iow_now:Nx \g_totoc_iow { \iow_char:N \\thispapertitle \iow_char:N \{ \exp_not:V \l_title_clean_tl \iow_char:N \} }

}

\RenewDocumentCommand{\author}{ O{} m m }
{
	\tl_set:Nn \l_author_tl {#2}
	\tl_set:Nn \l_author_email_tl {#1}
	\tl_set:Nn \l_author_affiliation_tl {#3}
	\seq_gput_right:Nn \g_authors_seq {#2\textsuperscript{#3}
		\tl_if_empty:nTF {#1} {} {\textsuperscript{,}\thanks{#1}}
	}
	\iow_now:Nx \g_totoc_iow { \iow_char:N \\thispaperauthor \iow_char:N \{ \exp_not:n  {#2} \iow_char:N \} }

}

\NewDocumentCommand{\affiliation}{m  m}{
	\tl_set:Nn \l_affiliation_tl {#1}
	\tl_set:Nn \l_affiliation_number_tl {#2}
	\seq_gput_right:Nn \g_affiliation_seq {\textsuperscript{#2}\ #1}
}

\NewDocumentCommand{\@affiliation}{}{
	\seq_use:Nnnn \g_affiliation_seq  {\\[2ex] }{ \\[2ex] }{ \\[2ex] }
}

\NewDocumentCommand{\authors}{}
{
	\seq_use:Nnnn \g_authors_seq {,\ }{,\ }{,\ }
}

\NewDocumentCommand{\udc}{ m }
{
	\tl_set:Nn \l_udc_tl {#1}
}

\NewDocumentCommand{\pacs}{ m }
{
	\tl_set:Nn \l_pacs_tl {#1}
}

\NewDocumentCommand{\keywords}{m}
{
	\tl_set:Nn \l_keywords_tl { #1 }
}

\RenewDocumentCommand{\abstract}{ +m }
{
	\tl_set:Nn \l_abstract_tl {#1}
}

\NewDocumentCommand{\@abstract}{}
{
	\tl_use:N \l_abstract_tl
}


%%% --------------------------------------------------------
%%> \subsection{Make Title self}
%%% --------------------------------------------------------

\date{}
\RenewDocumentCommand{\@maketitle}{}{%
	\newpage
	\tl_if_empty:NTF \l_pacs_tl {} {\pacsnum\ \tl_use:N \l_pacs_tl}
	\tl_if_empty:NTF \l_udc_tl {} {\udcnum\ \tl_use:N \l_udc_tl}
	\nopagebreak
	\null
	\vskip 1em%
	\begin{center}%
		\let\footnote\thanks
		{\Large\bfseries\sffamily \text_uppercase:n {\tl_use:N \l_title_tl}\par}%
		\vskip 1.5em%
			{\large
				\lineskip .5em%
				\begin{center}%
					\authors
				\end{center}\par}%
		\vskip 1ex%
			{\@affiliation}%
	\end{center}%
}

\renewcommand\maketitle{\par%
\begingroup
\iflanguage{ukrainian}%
{\renewcommand\thefootnote{\itshape\asbuk{footnote}}}%
{%
	\iflanguage{russian}%
    {\renewcommand\thefootnote{\itshape\asbuk{footnote}}}%
    {%
		\iflanguage{english}{\renewcommand\thefootnote{\itshape\alph{footnote}}}%
        {%
        }%
	}%
}%
\if@twocolumn
	\ifnum \col@number=\@ne
		\@maketitle
		\noindent\hrulefill\vspace*{-2ex}
		\section*{\abstractname}
		\begingroup
                \quotation
                    \@abstract
                \endquotation
		\endgroup
		\noindent{\bfseries \keyname}:  ~  \tl_use:N \l_keywords_tl \par
		\smallskip\noindent\hrulefill\medskip
	\else
		\twocolumn[\@maketitle
			\noindent\hrulefill\vspace*{-2ex}
			\section*{\abstractname}
			\begingroup
                \quotation
                    \@abstract
        		\endquotation
			\endgroup
			\vskip1em
			\noindent{\bfseries \keyname}:  ~  \tl_use:N \l_keywords_tl \par
			\smallskip\noindent\hrulefill\bigskip
		]%
	\fi
\else
	\newpage
	\global\@topnum\z@   % Prevents figures from going at top of page.
	\@maketitle
\fi
\@thanks
\endgroup
\setcounter{footnote}{0}%
\global\let\thanks\relax
\global\let\maketitle\relax
\global\let\@maketitle\relax
\global\let\@thanks\@empty
\global\let\@author\@empty
\global\let\@date\@empty
\global\let\@title\@empty
\global\let\title\relax
\global\let\author\relax
\global\let\date\relax
\global\let\and\relax
}


%%% --------------------------------------------------------
%%> \section{Sections}
%%% --------------------------------------------------------

\def\@seccntformat#1{\csname the#1\endcsname.\enskip}
\renewcommand\section{\@startsection {section}{1}{\z@}%
    {-2ex \@plus -1ex \@minus -.2ex}%
    {1.3ex \@plus.2ex}%
    {\normalfont\large\bfseries\raggedright}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
    {-3.25ex\@plus -1ex \@minus -.2ex}%
    {1.5ex \@plus .2ex}%
    {\normalfont\bfseries\raggedright}}


%%% --------------------------------------------------------
%%> \section{Pictures and Tables}
%%% --------------------------------------------------------

\RequirePackage[labelformat=simple]{subcaption}
\RequirePackage[labelsep=period]{caption}
    \def\tabcaption{\def\@captype{table}\caption}
    \def\figcaption{\def\@captype{figure}\caption}
    \renewcommand\thesubfigure{(\alph{subfigure})}
    \captionsetup[subfigure]{justification=centering, subrefformat=parens}


%%% --------------------------------------------------------
%%> \section{Hyperlinks}
%%% --------------------------------------------------------

\RequirePackage[colorlinks=true,
	urlcolor = blue, %Colour for external hyperlinks
	linkcolor  = red!60!black, %Colour of internal links
	citecolor  = green!50!black, %Colour of citations
	bookmarks,
	bookmarksnumbered=true,
	breaklinks=true,
	unicode,
	linktoc = all,
	hypertexnames=false,
]{hyperref}

\endinput
