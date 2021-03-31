
Shader "Skybox/GalaxyNavigator"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #define UNITY_PASS_FORWARDBASE
            #include "UnityCG.cginc"
            #pragma multi_compile_fwdbase_fullshadows
            #pragma only_renderers d3d9 d3d11 glcore gles n3ds wiiu 
            #pragma target 3.0

            // make fog work
            //# //pragma multi_compile_fog

            #include "UnityCG.cginc"

            uniform sampler2D _MainTex; 

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f {
                float4 uv : TEXCOORD0;         //posWorld
                float4 vertex : SV_POSITION;   //pos
            };

            v2f vert (appdata v) {
                appdata v2;
                v2.vertex = v.vertex; //mul(v.vertex ,float4x4(-1,0.,0.,0.,  0.,1.,0.,0.,  0.,0.,1.,0.,  0.,0.,0.,1.));
                v2f o = (v2f)0;
                o.uv = mul(unity_ObjectToWorld, v2.vertex);
                o.vertex = UnityObjectToClipPos( v.vertex); // * float4(1.0,1.,1.0,1.) ); // squish skybox sphere
                return o;
            }

// Galaxy shader
//
// Created by Frank Hugenroth  /frankenburgh/   07/2015
// Released at nordlicht/bremen 2015

#define SCREEN_EFFECT 0

// random/hash function              
float hash( float n )
{
  return frac(cos(n)*41415.92653);
}

// 2d noise function
float noise( in float2 x )
{
  float2 p  = floor(x);
  float2 f  = smoothstep(0.0, 1.0, frac(x));
  float n = p.x + p.y*57.0;

  return lerp(lerp( hash(n+  0.0), hash(n+  1.0),f.x),
    lerp( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y);
}

float noise( in float3 x )
{
  float3 p  = floor(x);
  float3 f  = smoothstep(0.0, 1.0, frac(x));
  float n = p.x + p.y*57.0 + 113.0*p.z;

  return lerp(lerp(lerp( hash(n+  0.0), hash(n+  1.0),f.x),
    lerp( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y),
    lerp(lerp( hash(n+113.0), hash(n+114.0),f.x),
    lerp( hash(n+170.0), hash(n+171.0),f.x),f.y),f.z);
}

float3x3 m = float3x3( 0.00,  1.60,  1.20, -1.60,  0.72, -0.96, -1.20, -0.96,  1.28 );

// fractional Brownian motion
float fbmslow( float3 p )
{
  float f = 0.5000*noise( p ); p = mul(m,p*1.2);
  f += 0.2500*noise( p ); p = mul(m,p*1.3);
  f += 0.1666*noise( p ); p = mul(m,p*1.4);
  f += 0.0834*noise( p ); p = mul(m,p*1.84);
  return f;
}

float fbm( float3 p )
{
  float f = 0., a = 1., s=0.;
  f += a*noise( p ); p = mul(m,p*1.149); s += a; a *= .75;
  f += a*noise( p ); p = mul(m,p*1.41); s += a; a *= .75;
  f += a*noise( p ); p = mul(m,p*1.51); s += a; a *= .65;
  f += a*noise( p ); p = mul(m,p*1.21); s += a; a *= .35;
  f += a*noise( p ); p = mul(m,p*1.41); s += a; a *= .75;
  f += a*noise( p ); 
  return f/s;
}


         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection*.5;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz;                                             // ray origin

  float time = _Time.y * 0.1;

  float2 xy = -1.0 + 2.0*fragCoord.xy/float2(800,450); // / iResolution.xy;

  // fade in (1=10sec), out after 8=80sec;
  float fade = min(1., time*1.)*min(1.,max(0., 15.-time));
  // start glow after 5=50sec
  float fade2= max(0., time-10.)*0.37;
  float glow = max(-.25,1.+pow(fade2, 10.) - 0.001*pow(fade2, 25.));
  
  
  // get camera position and view direction
  float3 campos = float3(500.0, 850., -.0-cos((time-1.4)/2.)*2000.); // moving
  float3 camtar = float3(0., 0., 0.);
  
  float roll = 0.34;
  float3 cw = normalize(camtar-campos);
  float3 cp = float3(sin(roll), cos(roll),0.0);
  float3 cu = normalize(cross(cw,cp));
  float3 cv = normalize(cross(cu,cw));
  //float3 
//  rd = normalize( xy.x*cu + xy.y*cv + 1.6*cw );


  float3 light   = normalize( float3(  0., 0.,  0. )-campos );
  float sundot = clamp(dot(light,rd),0.0,1.0);

  // render sky

    // galaxy center glow
    float3 col = glow*1.2*min(float3(1.0, 1.0, 1.0), float3(2.0,1.0,0.5)*pow( sundot, 100.0 ));
    // moon haze
    col += 0.3*float3(0.8,0.9,1.2)*pow( sundot, 8.0 );

  // stars
  float st = pow(fbmslow(rd.zxy*440.3), 8.0);
  float sc = pow(fbmslow(rd.xyz*312.0), 7.0);
  float3 stars = 85.5*float3(sc,sc,sc)*float3(st,st,st);
  
  // moving background fog
    float3 cpos = 1500.*rd + float3(831.0-time*30., 321.0, 1000.0);
    col += float3(0.4, 0.5, 1.0) * ((fbmslow( cpos*0.0035 ) - .5));

  cpos += float3(831.0-time*33., 321.0, 999.);
    col += float3(0.6, 0.3, 0.6) * 10.0*pow((fbmslow( cpos*0.0045 )), 10.0);

  cpos += float3(3831.0-time*39., 221.0, 999.0);
    col += 0.03*float3(0.6, 0.0, 0.0) * 10.0*pow((fbmslow( cpos*0.0145 )), 2.0);

  // stars
  cpos = 1500.*rd + float3(831.0, 321.0, 999.);
  col += stars*fbm(cpos*0.0021);
  
  
  // Clouds
    float2 shift = float2( time*100.0, time*180.0 );
    float4 sum = float4(0,0,0,0); 
    float c = campos.y / rd.y; // cloud height
    float3 cpos2 = campos - c*rd;
    float radius = length(cpos2.xz)/1000.0;

    if (radius<1.8)
    {
      for (int q=10; q>-10; q--) // layers
      {
    if (sum.w>0.999) continue;
        float c = (float(q)*8.-campos.y) / rd.y; // cloud height
        float3 cpos = campos + c*rd;

      float see = dot(normalize(cpos), normalize(campos));
    float3 lightUnvis = float3(.0,.0,.0 );
    float3 lightVis   = float3(1.3,1.2,1.2 );
    float3 shine = lerp(lightVis, lightUnvis, smoothstep(0.0, 1.0, see));

    // border
      float radius = length(cpos.xz)/999.;
      if (radius>1.0)
        continue;

    float rot = 3.00*(radius)-time;
        cpos.xz = mul(cpos.xz,float2x2(cos(rot), -sin(rot), sin(rot), cos(rot)));
  
    cpos += float3(831.0+shift.x, 321.0+float(q)*lerp(250.0, 50.0, radius)-shift.x*0.2, 1330.0+shift.y); // cloud position
    cpos *= lerp(0.0025, 0.0028, radius); // zoom
        float alpha = smoothstep(0.50, 1.0, fbm( cpos )); // fracal cloud density
      alpha *= 1.3*pow(smoothstep(1.0, 0.0, radius), 0.3); // fade out disc at edges
      float3 dustcolor = lerp(float3( 2.0, 1.3, 1.0 ), float3( 0.1,0.2,0.3 ), pow(radius, .5));
        float3 localcolor = lerp(dustcolor, shine, alpha); // density color white->gray
      
    float gstar = 2.*pow(noise( cpos*21.40 ), 22.0);
    float gstar2= 3.*pow(noise( cpos*26.55 ), 34.0);
    float gholes= 1.*pow(noise( cpos*11.55 ), 14.0);
    localcolor += float3(1.0, 0.6, 0.3)*gstar;
    localcolor += float3(1.0, 1.0, 0.7)*gstar2;
    localcolor -= gholes;
      
        alpha = (1.0-sum.w)*alpha; // alpha/density saturation (the more a cloud layer\\\'s density, the more the higher layers will be hidden)
        sum += float4(localcolor*alpha, alpha); // sum up weightened color
    }
    
      for (int q=0; q<20; q++) // 120 layers
      {
    if (sum.w>0.999) continue;
        float c = (float(q)*4.-campos.y) / rd.y; // cloud height
        float3 cpos = campos + c*rd;

      float see = dot(normalize(cpos), normalize(campos));
    float3 lightUnvis = float3(.0,.0,.0 );
    float3 lightVis   = float3(1.3,1.2,1.2 );
    float3 shine = lerp(lightVis, lightUnvis, smoothstep(0.0, 1.0, see));

    // border
      float radius = length(cpos.xz)/200.0;
      if (radius>1.0)
        continue;

    float rot = 3.2*(radius)-time*1.1;
        cpos.xz = mul(cpos.xz,float2x2(cos(rot), -sin(rot), sin(rot), cos(rot)));
  
    cpos += float3(831.0+shift.x, 321.0+float(q)*lerp(250.0, 50.0, radius)-shift.x*0.2, 1330.0+shift.y); // cloud position
        float alpha = 0.1+smoothstep(0.6, 1.0, fbm( cpos )); // fracal cloud density
      alpha *= 1.2*(pow(smoothstep(1.0, 0.0, radius), 0.72) - pow(smoothstep(1.0, 0.0, radius*1.875), 0.2)); // fade out disc at edges
        float3 localcolor = float3(0.0, 0.0, 0.0); // density color white->gray
  
        alpha = (1.0-sum.w)*alpha; // alpha/density saturation (the more a cloud layer\\\'s density, the more the higher layers will be hidden)
        sum += float4(localcolor*alpha, alpha); // sum up weightened color
    }
    }
  float alpha = smoothstep(1.-radius*.5, 1.0, sum.w);
    sum.rgb /= sum.w+0.0001;
    sum.rgb -= 0.2*float3(0.8, 0.75, 0.7) * pow(sundot,10.0)*alpha;
    sum.rgb += min(glow, 10.0)*0.2*float3(1.2, 1.2, 1.2) * pow(sundot,5.0)*(1.0-alpha);

    col = lerp( col, sum.rgb , sum.w);//*pow(sundot,10.0) );

    // haze
//  col = fade*lerp(col, float3(0.3,0.5,.9), 29.0*(pow( sundot, 50.0 )-pow( sundot, 60.0 ))/(2.+9.*abs(rd.y)));

#if SCREEN_EFFECT == 1
    if (time<2.5)
    {
      // screen effect
      float c = (col.r+col.g+col.b)* .3 * (.6+.3*cos(fragCoord.y*1.2543)) + .1*(noise((xy+time*2.)*294.)*noise((xy-time*3.)*321.));
        c += max(0.,.08*sin(10.*time+xy.y*7.2543));
        // flicker
    col = float3(c, c, c) * (1.-0.5*pow(noise(float2(time*99., 0.)), 9.));
    }
    else
    {
        // bam
        float c = clamp(1.-(time-2.5)*6., 0., 1. );
        col = lerp(col, float3(1.,1.,1.),c);
    }
#endif
    
    // Vignetting
//  float2 xy2 = gl_FragCoord.xy / iResolution.xy;
//  col *= float3(.5, .5, .5) + 0.25*pow(100.0*xy2.x*xy2.y*(1.0-xy2.x)*(1.0-xy2.y), .5 ); 

  fragColor = float4(col,1.0);


                return fragColor;
            }

            ENDCG
        }
    }
}








