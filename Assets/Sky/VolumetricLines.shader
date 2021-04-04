
Shader "Skybox/VolumetricLines"
{
    Properties
    {
        _Sound ("Texture", 2D) = "white" {}
         _Beat ("_Beat", float) = 0.0
         _Volume ("_Volume", float) = 0.0
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
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            uniform sampler2D _Sound; 
            uniform int _SoundArrayLength = 256;
            float _Beat;
            uniform float _SoundArray[256];
            uniform float _Volume;

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



float3 hash3( float n )
{
    return frac(sin(float3(n,n+1.0,n+2.0))*float3(43758.5453123,22578.1459123,19642.3490423));
}

float3 snoise3( in float x )
{
    float p = floor(x);
    float f = frac(x);

    f = f*f*(3.0-2.0*f);

    return -1.0 + 2.0*lerp( hash3(p+0.0), hash3(p+1.0), f );
}

float freqs[16];

float dot2(in float3 v ) { return dot(v,v); }
float2 usqdLineSegment( float3 o, float3 d, float3 a, float3 b )
{
    float3 oa = o - a;
    float3 ob = o - b;
    float3 va = d*dot(oa,d)-oa;
    float3 vb = d*dot(ob,d)-ob;
    
    float3 v1 = va;
    float3 v2 = vb-v1;
    float h = clamp( -dot(v1,v2)/dot(v2,v2), 0.0, 1.0 );

    float di = dot2(v1+v2*h);
    
    return float2(di,h);
}

float3 castRay( float3 ro, float3 rd, float linesSpeed )
{
    float3 col = float3(0.0,0,0);
    
        
    float mindist = 10000.0;
    float3 p = float3(0.2,.2,.2);
    float h = 0.0;
    float rad = 0.04 + 0.15*freqs[0];
    float mint = 0.0;
    for( int i=0; i<128; i++ )
    {
        float3 op = p;
        
        op = p;
        p  = 1.25*1.0*normalize(snoise3( 64.0*h + linesSpeed*0.015*_Time.y ));
        
        float2 dis = usqdLineSegment( ro, rd, op, p );
        
        float3 lcol = 0.6 + 0.4*sin( 10.0*6.2831*h + float3(0.0,0.6,0.9) );
        
//        float m = pow( tex2D( _Sound, float2(h*0.5,0.25) ).x, 2.0 )*(1.0+2.0*h);
        float m = pow( min(.9,_SoundArray[floor(h*0.5)]*1/max(.01,_Volume)) , 2.0 )*(1.0+2.0*h);
        
        float f = 1.0 - 4.0*dis.y*(1.0-dis.y);
        float width = 1240.0 - 1000.0*f;
        width *= 0.25;
        float ff = 1.0*exp(-0.06*dis.y*dis.y*dis.y);
        ff *= m;
        col += 0.3*lcol*exp( -0.3*width*dis.x )*ff;
        col += 0.5*lcol*exp( -8.0*width*dis.x )*ff;
        h += 1.0/128.0;
    }


    return col;
}


         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_Sound, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz*.0001;                                             // ray origin


    float2 q = float2(1,1); //fragCoord.xy/iResolution.xy;
//    float2 p = -1.0+2.0*q;
//    p.x *= iResolution.x/iResolution.y;
//    float2 mo = iMouse.xy/iResolution.xy;
         
    float time = _Time.y;


    for( int i=0; i<16; i++ )
        freqs[i] = clamp( 1.9*pow( tex2D( _Sound, float2( 0.05 + 0.5*float(i)/16.0, 0.25 ) ).x, 3.0 ), 0.0, 1.0 );
    
    // camera   
    float3 ta = float3( 0.0, 0.0, 0.0 );

float iChannelTime = 1.0+_Time.y;
float isFast = smoothstep( 35.8, 35.81, iChannelTime );
    isFast  -= smoothstep( 61.8, 61.81, iChannelTime );
    isFast  += smoothstep( 78.0, 78.01, iChannelTime );
    isFast  -= smoothstep(103.0,103.01, iChannelTime );
    isFast  += smoothstep(140.0,140.01, iChannelTime );
    isFast  -= smoothstep(204.0,204.01, iChannelTime );
    
    float camSpeed = 1.0 + 40.0*isFast; 


    float beat = 0.0; //floor( max((iChannelTime[0]-35.7+0.4)/0.81,0.0) );
    time += beat*10.0*isFast;
    camSpeed *= lerp( 1.0, sign(sin( beat*1.0 )), isFast );

    
float linesSpeed =  smoothstep( 22.7, 22.71, iChannelTime ); 
      linesSpeed -= smoothstep( 61.8, 61.81, iChannelTime );
      linesSpeed += smoothstep( 78.0, 78.01, iChannelTime );
      linesSpeed -= smoothstep(140.0,140.01, iChannelTime );

    
    ta  = 0.2*float3( cos(0.1*time), 0.0*sin(0.1*time), sin(0.07*time) );

//    float3 ro = float3( 1.0*cos(camSpeed*0.05*time+6.28*mo.x), 0.0, 1.0*sin(camSpeed*0.05*time+6.2831*mo.x) );
    float roll = 0.25*sin(camSpeed*0.01*time);
    
    // camera tx
    float3 cw = normalize( ta-ro );
    float3 cp = float3( sin(roll), cos(roll),0.0 );
    float3 cu = normalize( cross(cw,cp) );
    float3 cv = normalize( cross(cu,cw) );
    //float3 rd = normalize( p.x*cu + p.y*cv + 1.2*cw );

    float curve  = smoothstep( 61.8, 71.0, iChannelTime );
          curve -= smoothstep(103.0,113.0, iChannelTime );
    rd.xy += curve*0.025*float2( sin(34.0*q.y), cos(34.0*q.x) );
    rd = normalize(rd);
    
    
    ro *= 1.0 - linesSpeed*0.5*freqs[1];
    float3 col = castRay( ro, rd, 1.0 + 20.0*linesSpeed );
    col = col*col*2.4;
    


    // fade to black
    col *= 1.0 - smoothstep(218.0,228.00, iChannelTime );
    col *=       smoothstep(  0.0,  4.00, iChannelTime );
    if( iChannelTime>61.8 && iChannelTime<65.0 )
    col *= float3(1.0,1,1)*clamp( (iChannelTime-61.8)/(65.0-61.8), 0.0, 1.0 );
   

    col *= 0.15+0.85*pow( 16.0*q.x*q.y*(1.0-q.x)*(1.0-q.y), 0.15 );

    fragColor=float4( col, 1.0 );
                return fragColor;
            }

            ENDCG
        }
    }
}



